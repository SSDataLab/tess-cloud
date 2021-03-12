"""Defines the `TessImage` and `TessImageList` classes.
"""
import asyncio
import io
import re
import struct
import warnings
from collections import UserList
from functools import lru_cache
from typing import Union
import contextlib

import aiohttp
import aioboto3
from astropy.io.fits import file
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from botocore import UNSIGNED
from botocore.config import Config
from pandas import DataFrame
import tqdm

from . import MAX_CONCURRENT_DOWNLOADS, TESS_S3_BUCKET, log
from .manifest import get_s3_uri, _load_ffi_manifest
from .targetpixelfile import TargetPixelFile

# FITS standard specifies that header and data units
# shall be a multiple of 2880 bytes long.
FITS_BLOCK_SIZE = 2880  # bytes

# TESS FFI dimensions
FFI_COLUMNS = 2136  # i.e. NAXIS1
FFI_ROWS = 2078  # i.e. NAXIS2

BYTES_PER_PIX = 4  # float32

FFI_FILENAME_REGEX = r".*-s(\d+)-(\d)-(\d)-.*"
MAST_FFI_URL_PREFIX = (
    "https://mast.stsci.edu/portal/Download/file?uri=mast:TESS/product/"
)


class TessImage:
    """TESS FFI image hosted at AWS S3.

    Parameters
    ----------
    url : str
        URL or filename of a TESS FFI image on AWS S3.
    """

    def __init__(self, url, data_ext=None, data_offset=None):
        if "/" in url:
            self.filename = url.split("/")[-1]
            self._url = url
        else:
            self.filename = url
            self._url = None

        if data_ext is None:
            if "hlsp_tica" in url:
                data_ext = 0
            else:
                data_ext = 1
        self.data_ext = data_ext

        if data_offset is None and "hlsp_tica" in url:
            self._data_offset = 17280
        elif data_offset:
            self._data_offset = data_offset

    def __repr__(self):
        return f'TessImage("{self.filename}")'

    def _parse_filename(self) -> dict:
        """Extracts sector, camera, ccd from a TESS FFI filename."""
        search = re.search(FFI_FILENAME_REGEX, self.filename)
        if not search:
            raise ValueError(f"Unrecognized FFI filename: {self.filename}")
        return {
            "sector": int(search.group(1)),
            "camera": int(search.group(2)),
            "ccd": int(search.group(3)),
        }

    @property
    def sector(self) -> int:
        return self._parse_filename()["sector"]

    @property
    def camera(self) -> int:
        return self._parse_filename()["camera"]

    @property
    def ccd(self) -> int:
        return self._parse_filename()["ccd"]

    @property
    def url_mast(self) -> str:
        """Returns the URL for the image at MAST."""
        return MAST_FFI_URL_PREFIX + self.filename

    @property
    def url(self) -> str:
        """Returns the URL for the image at AWS S3."""
        if not self._url:
            self._url = get_s3_uri(self.filename)
        return self._url

    @lru_cache
    def _get_s3_key(self) -> str:
        return self.url.split(f"{TESS_S3_BUCKET}/")[1]

    @property
    def data_offset(self):
        if not hasattr(self, "_data_offset"):
            self._data_offset = self._find_data_offset(ext=self.data_ext)
        return self._data_offset

    async def _async_read_block_http(
        self, offset: int = None, length: int = None, client=None
    ):
        headers = {}
        if not (offset is None or length is None):
            headers["Range"] = f"bytes={offset}-{offset+length-1}"

        if client:
            async with client.get(self.url, headers=headers) as resp:
                return await resp.read()

        # Making a new client for every request is slow; avoid if possible!
        async with aiohttp.ClientSession() as client:
            async with client.get(self.url, headers=headers) as resp:
                return await resp.read()

    async def _async_read_block_s3(
        self, offset: int = None, length: int = None, client=None
    ):
        if length is None or offset is None:
            byterange = ""
        else:
            byterange = f"bytes={offset}-{offset+length-1}"

        if client:
            resp = await client.get_object(
                Bucket=TESS_S3_BUCKET, Key=self._get_s3_key(), Range=byterange
            )
            return await resp["Body"].read()

        # Making a new client for every request is slow; avoid if possible!
        async with _default_s3client() as client:
            resp = await client.get_object(
                Bucket=TESS_S3_BUCKET, Key=self._get_s3_key(), Range=byterange
            )
            return await resp["Body"].read()

    async def async_read_block(
        self, offset: int = None, length: int = None, client=None
    ) -> bytes:
        """Read a block of bytes.

        Parameters
        ----------
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read
        """
        # Use Semaphore to limit the number of concurrent downloads
        async with MAX_CONCURRENT_DOWNLOADS:
            if self.url[:2] == "s3":
                return await self._async_read_block_s3(offset, length, client)
            elif self.url[:4] == "http":
                return await self._async_read_block_http(offset, length, client)
            else:
                raise ValueError("url must start with http:// or s3://")

    def read_block(self, offset: int = None, length: int = None) -> bytes:
        """Read a block of bytes from AWS S3.

        Parameters
        ----------
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read
        """
        return _sync_call(self.async_read_block, offset=offset, length=length)

    def read_header(self, ext: int = None) -> fits.Header:
        if ext is None:
            ext = self.data_ext
        content = self.read_block(0, self.data_offset)
        # Open the file and extract the fits header
        with warnings.catch_warnings():
            # Ignore "File may have been truncated" warning
            warnings.simplefilter("ignore", AstropyUserWarning)
            hdr = fits.getheader(io.BytesIO(content), ext=ext)
        return hdr

    def read_wcs(self, ext: int = None) -> WCS:
        """Downloads the image WCS."""
        if ext is None:
            ext = self.data_ext
        return WCS(self.read_header(ext=ext))

    async def async_read(self) -> fits.HDUList:
        """Open the entire image as an AstroPy HDUList object."""
        return fits.open(io.BytesIO(await self.async_read_block()))

    def read(self) -> fits.HDUList:
        """Open the entire image as an AstroPy HDUList object."""
        return _sync_call(self.async_read)

    def _find_data_offset(self, ext: int = None) -> int:
        """Returns the byte offset of the start of the data section."""
        if ext is None:
            ext = self.data_ext
        # We'll assume the data starts within the first 10 FITS BLOCKs.
        # This means the method will currently only work for extensions 0 and 1 of a TESS FFI file.
        max_seek = FITS_BLOCK_SIZE * 12
        data = self.read_block(0, max_seek)
        current_ext = 0
        offset = 0
        while offset <= max_seek:
            block = data[offset : offset + FITS_BLOCK_SIZE]
            offset += FITS_BLOCK_SIZE
            # Header sections end with "END" followed by whitespace until the end of the block
            if re.search("END\s*$", block.decode("ascii")):
                if current_ext == ext:
                    log.debug(f"data_offset={offset} for {self.url}")
                    return offset
                current_ext += 1
        return None

    def _find_pixel_offset(self, col: int, row: int) -> int:
        """Returns the byte offset of a specific pixel position."""
        pixel_offset = col + row * FFI_COLUMNS
        return self.data_offset + BYTES_PER_PIX * pixel_offset

    def _find_pixel_blocks(self, col: int, row: int, shape=(1, 1)) -> list:
        """Returns the byte ranges of a rectangle."""
        result = []
        col1 = int(col) - shape[0] // 2
        row1 = int(row) - shape[1] // 2

        if col1 < 0 or col1 >= FFI_COLUMNS:
            raise ValueError(
                f"column out of bounds (col must be in range 0-{FFI_COLUMNS})"
            )
        if row1 < 0 or row1 >= FFI_ROWS:
            raise ValueError(f"row out of bounds (row must be in range 0-{FFI_ROWS})")

        for myrow in range(row1, row1 + shape[1]):
            begin = self._find_pixel_offset(col1, myrow)
            end = self._find_pixel_offset(col1 + shape[0], myrow)
            myrange = (
                begin,
                end - begin,
            )
            result.append(myrange)
        return result

    async def _async_cutout_array(
        self, col: int, row: int, shape=(5, 5), client=None
    ) -> np.array:
        """Returns a 2D array of pixel values."""
        blocks = self._find_pixel_blocks(col=col, row=row, shape=shape)
        bytedata = await asyncio.gather(
            *[
                self.async_read_block(offset=blk[0], length=blk[1], client=client)
                for blk in blocks
            ]
        )
        data = []
        for b in bytedata:
            n_pixels = len(b) // BYTES_PER_PIX
            values = struct.unpack(">" + "f" * n_pixels, b)
            data.append(values)
        return np.array(data)

    async def async_cutout(
        self, col: int, row: int, shape=(5, 5), client=None
    ) -> "Cutout":
        """Returns a cutout."""
        flux = await self._async_cutout_array(
            col=col, row=row, shape=shape, client=client
        )
        time = 0
        cadenceno = 0
        quality = 0
        flux_err = flux.copy()
        flux_err[:] = np.nan
        return Cutout(
            time=time,
            cadenceno=cadenceno,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
        )

    def cutout(self, col, row, shape=(5, 5)) -> "Cutout":
        """Returns a cutout."""
        return _sync_call(self.async_cutout, col=col, row=row, shape=shape)


class TessImageList(UserList):
    def __repr__(self):
        x = []
        if len(self) > 8:
            show = [0, 1, 2, 3, -4, -3, -2, -1]
        else:
            show = range(len(self))
        for idx in show:
            x.append(str(self[idx]))
        if len(self) > 8:
            x.insert(4, "...")
        return f"List of {len(self)} images\n ↳[" + "\n   ".join(x) + "]"

    def to_pandas(self) -> DataFrame:
        data = [
            {
                "filename": im.filename,
                "sector": im.sector,
                "camera": im.camera,
                "ccd": im.ccd,
                "url_mast": im.url_mast,
            }
            for im in self
        ]
        return DataFrame(data)

    @classmethod
    def from_catalog(cls, catalog: DataFrame):
        # We use raw=True because it gains significant speed
        series = catalog.apply(
            lambda x: TessImage(filename=x[0], begin=x[4], end=x[5]), axis=1, raw=True
        )
        return cls(series.values)

    async def _get_cutouts(self, col, row, shape):
        async with aiohttp.ClientSession() as client:
            # Create list of functions to be executed
            flist = [
                img.async_cutout(col=col, row=row, shape=shape, client=client)
                for img in self
            ]
            # Create tasks for the sake of allowing a progress bar to be shown.
            # We'd want to use `asyncio.gather(*flist)` here to obtain the results in order,
            # but the progress bar needs `asyncio.as_completed` to work.
            tasks = [asyncio.create_task(f) for f in flist]
            for t in tqdm.tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Downloading cutouts",
            ):
                await t
            # Now take care of getting the results in order
            results = [t.result() for t in tasks]
            return results

    def cutout(self, col: int, row: int, shape=(5, 5)):
        cutouts = asyncio.run(self._get_cutouts(col=col, row=row, shape=shape))
        tpf = TargetPixelFile.from_cutouts(cutouts)
        return tpf.to_lightkurve()


class Cutout:
    def __init__(
        self,
        time: float,
        cadenceno: int,
        flux: np.ndarray,
        flux_err: np.ndarray,
        quality: int,
        meta: dict = None,
    ):
        self.time = time
        self.cadenceno = cadenceno
        self.flux = flux
        self.flux_err = flux_err
        self.quality = quality
        self.meta = meta


def _default_s3client():
    return aioboto3.client("s3", config=Config(signature_version=UNSIGNED))


def _sync_call(func, *args, **kwargs):
    return asyncio.run(func(*args, **kwargs))


def _data_offset_lookup(camera=1, ccd=1):
    """Returns a lookup table mapping sector => data offset."""
    from . import manifest

    lookup = {}
    sector = 1
    df = manifest._load_manifest_table()
    while True:
        sector_ffis = df[
            df.path.str.match(
                f"tess/public/ffi/s{sector:04d}/.*s{sector:04d}-{camera}-{ccd}.*ffic.fits"
            )
        ]
        if len(sector_ffis) == 0:
            break
        uri = sector_ffis.path.iloc[200]
        offset = TessImage(uri)._find_data_offset()
        lookup[sector] = offset
        print(f"{sector}: {offset} {uri}")
        sector += 1
    return lookup


def list_images(sector: int, camera: int = None, ccd: int = None):
    """Returns a list of the FFIs for a given sector/camera/ccd."""
    if camera is None:
        camera = "\d"  # regex
    if ccd is None:
        ccd = "\d"  # regex
    ffi_files = _load_ffi_manifest()
    mask = ffi_files.path.str.match(
        f".*tess(\d+)-s{sector:04d}-{camera}-{ccd}-\d+-._ffic.fits"
    )
    return TessImageList(
        [TessImage("s3://stpubdata/" + x) for x in ffi_files[mask].path.values]
    )
