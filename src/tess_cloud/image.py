"""Defines the `TessImage` and `TessImageList` classes.
"""
import asyncio
import io
import re
import struct
import warnings
from functools import lru_cache

import aiohttp
import aioboto3
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.time import Time
from botocore import UNSIGNED
from botocore.config import Config
import backoff

from . import MAX_CONCURRENT_DOWNLOADS, MAX_CONCURRENT_CUTOUTS, TESS_S3_BUCKET, log
from .manifest import get_s3_uri

# FITS standard specifies that header and data units
# shall be a multiple of 2880 bytes long.
FITS_BLOCK_SIZE = 2880  # bytes

# TESS FFI dimensions
FFI_COLUMNS = 2136  # i.e. NAXIS1
FFI_ROWS = 2078  # i.e. NAXIS2

BYTES_PER_PIX = 4  # float32

FFI_FILENAME_REGEX = r".*-s(\d+)-(\d)-(\d)-.*"


class TessImage:
    """TESS FFI image hosted at AWS S3.

    Parameters
    ----------
    url : str
        URL or filename of a TESS FFI image on AWS S3.
    """

    def __init__(
        self,
        url,
        data_ext=None,
        data_offset=None,
        meta=None,
    ):
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

        self.data_offset = data_offset

        if meta is None:
            self.meta = {}
        else:
            self.meta = meta

    def __repr__(self):
        return f'TessImage("{self.filename}")'

    @property
    def _client_type(self) -> str:
        if self.url[:2] == "s3" or self.url[:9] == "stpubdata":
            return "s3"
        elif self.url[:4] == "http":
            return "http"
        else:
            raise ValueError("url must start with https:// or s3://")

    def _get_default_client(self):
        if self._client_type == "s3":
            return _default_s3_client()
        else:
            return _default_http_client()

    @property
    def sector(self) -> int:
        return self.meta.get("sector")

    @property
    def camera(self) -> int:
        return self.meta.get("camera")

    @property
    def ccd(self) -> int:
        return self.meta.get("ccd")

    @property
    def time(self) -> str:
        return self.meta.get("time")

    @property
    def cadenceno(self) -> int:
        return self.meta.get("cadenceno")

    @property
    def quality(self) -> int:
        return self.meta.get("quality")

    @property
    def url(self) -> str:
        """Returns the URL for the image at AWS S3."""
        if not self._url:
            self._url = get_s3_uri(self.filename)
        return self._url

    @lru_cache()
    def _get_s3_key(self) -> str:
        return self.url.split(f"{TESS_S3_BUCKET}/")[1]

    async def _async_read_block_http(
        self, offset: int = None, length: int = None, client=None
    ):
        headers = {}
        if not (offset is None or length is None):
            headers["Range"] = f"bytes={offset}-{offset+length-1}"

        if client:
            async with client.get(self.url, headers=headers) as resp:
                # TODO: consider checking resp.status == 206 here?
                return await resp.read()

        # Making a new client for every request is slow; avoid if possible!
        async with _default_http_client() as client:
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
        async with _default_s3_client() as client:
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
            if self._client_type == "s3":
                result = await self._async_read_block_s3(offset, length, client)
            elif self._client_type == "http":
                result = await self._async_read_block_http(offset, length, client)
            return result

    def read_block(self, offset: int = None, length: int = None) -> bytes:
        """Read a block of bytes from AWS S3.

        Parameters
        ----------
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read
        """
        result = _sync_call(self.async_read_block, offset=offset, length=length)
        return result

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

    async def _find_data_offset(
        self, ext: int = None, return_header: bool = False
    ) -> int:
        """Returns the byte offset of the start of the data section.

        Unfortunately we cannot assume that the data_offset is consistent
        across images of a similar series, because the number of WCS keywords
        tends to change.  For example, keyword "AP_0_6" is present in:
            https://archive.stsci.edu/hlsps/tica/s0035/cam1-ccd1/hlsp_tica_tess_ffi_s0035-o1-00149185-cam1-ccd1_tess_v01_img.fits
        but not in
            https://archive.stsci.edu/hlsps/tica/s0035/cam1-ccd1/hlsp_tica_tess_ffi_s0035-o1-00147989-cam1-ccd1_tess_v01_img.fits
        """
        if self.data_offset:
            return self.data_offset
        if ext is None:
            ext = self.data_ext
        # We'll assume the data starts within the first 10 FITS BLOCKs.
        # This means the method will currently only work for extensions 0 and 1 of a TESS FFI file.
        max_seek = FITS_BLOCK_SIZE * 12
        data = await self.async_read_block(0, max_seek)
        current_ext = 0
        offset = 0
        prev_offset = 0  # necessary to support the `return_header` feature
        while offset <= max_seek:
            block = data[offset : offset + FITS_BLOCK_SIZE]
            offset += FITS_BLOCK_SIZE
            # Header sections end with "END" followed by whitespace until the end of the block
            if re.search(r"END\s*$", block.decode("ascii")):
                if current_ext == ext:
                    log.debug(f"data_offset={offset} for {self.url}")
                    if self.data_ext == ext:
                        self.data_offset = offset
                    if return_header:
                        return offset, data[prev_offset:offset]
                    else:
                        return offset
                current_ext += 1
                prev_offset = offset
        return None

    async def _find_pixel_offset(self, column: int, row: int) -> int:
        """Returns the byte offset of a specific pixel position."""
        data_offset = await self._find_data_offset(ext=self.data_ext)
        pixel_offset = column + row * FFI_COLUMNS
        return data_offset + BYTES_PER_PIX * pixel_offset

    async def _find_pixel_blocks(self, column: int, row: int, shape=(1, 1)) -> list:
        """Returns the byte ranges of a rectangle."""
        result = []
        col1 = int(column) - shape[0] // 2
        row1 = int(row) - shape[1] // 2

        if col1 < 0 or col1 >= FFI_COLUMNS:
            raise ValueError(f"column out of bounds (must be in range 0-{FFI_COLUMNS})")
        if row1 < 0 or row1 >= FFI_ROWS:
            raise ValueError(f"row out of bounds (must be in range 0-{FFI_ROWS})")

        for myrow in range(row1, row1 + shape[1]):
            begin = await self._find_pixel_offset(col1, myrow)
            end = await self._find_pixel_offset(col1 + shape[0], myrow)
            myrange = (
                begin,
                end - begin,
            )
            result.append(myrange)
        return result

    @backoff.on_exception(
        backoff.constant,
        (aiohttp.ClientError, aiohttp.ClientConnectorError),
        interval=1,
        max_tries=3,
    )
    async def _async_cutout_array(
        self, column: int, row: int, shape=(5, 5), client=None
    ) -> np.array:
        """Returns a 2D array of pixel values."""
        blocks = await self._find_pixel_blocks(column=column, row=row, shape=shape)
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
        self, column: int, row: int, shape=(5, 5), client=None
    ) -> "Cutout":
        """Returns a cutout."""
        async with MAX_CONCURRENT_CUTOUTS:
            flux = await self._async_cutout_array(
                column=column, row=row, shape=shape, client=client
            )

        if self.time:
            time = Time(self.time).btjd
        else:
            time = self.time

        flux_err = flux.copy()
        flux_err[:] = np.nan
        return Cutout(
            time=time,
            flux=flux,
            flux_err=flux_err,
            sector=self.sector,
            camera=self.camera,
            ccd=self.ccd,
            column=column,
            row=row,
            cadenceno=self.cadenceno,
            quality=self.quality,
            meta=self.meta,
        )

    def cutout(self, column, row, shape=(5, 5)) -> "Cutout":
        """Returns a cutout."""
        return _sync_call(self.async_cutout, column=column, row=row, shape=shape)


class Cutout:
    def __init__(
        self,
        time: float,
        flux: np.ndarray,
        flux_err: np.ndarray,
        sector: int,
        camera: int,
        ccd: int,
        column: int,
        row: int,
        cadenceno: int,
        quality: int,
        meta: dict = None,
    ):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.column = column
        self.row = row
        self.cadenceno = cadenceno
        self.quality = quality
        self.meta = meta


def _default_http_client():
    return aiohttp.ClientSession()


def _default_s3_client():
    return aioboto3.client("s3", config=Config(signature_version=UNSIGNED))


def _sync_call(func, *args, **kwargs):
    return asyncio.run(func(*args, **kwargs))
