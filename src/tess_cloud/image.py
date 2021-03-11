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

import aioboto3
from astropy.io.fits import file
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from botocore import UNSIGNED
from botocore.config import Config
from pandas import DataFrame

from . import MAX_CONCURRENT_DOWNLOADS, TESS_S3_BUCKET, log
from .manifest import get_s3_uri

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

    def __init__(self, url, data_offset=None):
        if "/" in url:
            self.filename = url.split("/")[-1]
            self._url = url
        else:
            self.filename = url
            self._url = None

        if self.data_offset:
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

    @property
    def key(self) -> str:
        return self.url.split(f"{TESS_S3_BUCKET}/")[1]

    @property
    def data_offset(self):
        if not hasattr(self, "_data_offset") or self._data_offset is None:
            self._data_offset = self._find_data_offset(ext=1)
        return self._data_offset

    async def async_read_block(
        self, offset: int = None, length: int = None, s3client=None
    ) -> bytes:
        """Read a block of bytes from AWS S3.

        Parameters
        ----------
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read
        """
        if length is None or offset is None:
            byterange = ""
        else:
            byterange = f"bytes={offset}-{offset+length-1}"
        log.debug(f"reading {range} from {self.filename}")

        # Use Semaphore to limit the number of concurrent downloads
        async with MAX_CONCURRENT_DOWNLOADS:

            if s3client:
                resp = await s3client.get_object(
                    Bucket=TESS_S3_BUCKET, Key=self.key, Range=byterange
                )
                return await resp["Body"].read()

            async with _default_s3client() as s3client:
                resp = await s3client.get_object(
                    Bucket=TESS_S3_BUCKET, Key=self.key, Range=byterange
                )
                return await resp["Body"].read()

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

    def read_header(self, ext=1) -> fits.Header:
        content = self.read_block(0, self.data_offset)
        # Open the file and extract the fits header
        with warnings.catch_warnings():
            # Ignore "File may have been truncated" warning
            warnings.simplefilter("ignore", AstropyUserWarning)
            hdr = fits.getheader(io.BytesIO(content), ext=ext)
        return hdr

    def read_wcs(self) -> WCS:
        """Downloads the image WCS."""
        return WCS(self.read_header(ext=1))

    async def async_read(self) -> fits.HDUList:
        """Open the entire image as an AstroPy HDUList object."""
        return fits.open(io.BytesIO(await self.async_read_block()))

    def read(self) -> fits.HDUList:
        """Open the entire image as an AstroPy HDUList object."""
        return _sync_call(self.async_read)

    def _find_data_offset(self, ext: int = 1) -> int:
        """Returns the byte offset of the start of the data section."""
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
        self, col: int, row: int, shape=(5, 5), s3client=None
    ) -> np.array:
        """Returns a 2D array of pixel values."""
        blocks = self._find_pixel_blocks(col=col, row=row, shape=shape)
        bytedata = await asyncio.gather(
            *[
                self.async_read_block(offset=blk[0], length=blk[1], s3client=s3client)
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
        self, col: int, row: int, shape=(5, 5), s3client=None
    ) -> "Cutout":
        """Returns a cutout."""
        flux = await self._async_cutout_array(
            col=col, row=row, shape=shape, s3client=s3client
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
