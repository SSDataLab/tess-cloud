"""
Notes
-----
* Data offset for ext 1 is always 20160 bytes, EXCEPT if the SIP keywords are missing.


TODO
----


"""
import asyncio
import re
import struct

import boto3
import aioboto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np

from . import log


# The maximum number of downloads to await at any given time is controlled using a semaphore.
# Too many parallel downloads may lead to read timeouts on slow connections.
MAX_CONCURRENT_DOWNLOADS = asyncio.Semaphore(100)

# FITS standard specifies that header and data units
# shall be a multiple of 2880 bytes long.
FITS_BLOCK_SIZE = 2880  # bytes

# TESS FFI dimensions
FFI_COLUMNS = 2136  # i.e. NAXIS1
FFI_ROWS = 2078  # i.e. NAXIS2

BYTES_PER_PIX = 4  # float32

# S3FILESYSTEM = s3fs.S3FileSystem(anon=True)
# Use a small block size when reading from S3 to avoid wasting time on excessive buffering
S3_BLOCK_SIZE = 2880  # bytes


s3client = aioboto3.client("s3", config=Config(signature_version=UNSIGNED))


class TessImage:
    def __init__(self, url, data_offset=None, s3=None):
        self.url = url

        # Infer S3 key and bucket from url
        self.bucket = "stpubdata"
        if self.url.startswith("stpubdata") or self.url.startswith("s3://stpubdata"):
            self.key = self.url.split("stpubdata/")[1]
        else:
            self.key = self.url

        if data_offset:
            self._data_offset = data_offset

        self.s3 = s3

    @property
    def _s3_client(self):
        if not hasattr(self, "__s3_client"):
            self.__s3_client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            )
        return self.__s3_client

    @property
    def data_offset(self):
        if not hasattr(self, "_data_offset"):
            self._data_offset = self._find_data_offset(ext=1)
        return self._data_offset

    def read_header(self):
        return self.read_block(0, self.data_offset)

    async def async_read_block(self, offset: int, length: int) -> bytes:
        """Alternative async implementation using aioboto3.

        Parameters
        ----------
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read
        """
        # async with aioboto3.client("s3", config=Config(signature_version=UNSIGNED)) as s3:
        async with MAX_CONCURRENT_DOWNLOADS:  # Use Semaphore to limit the number of concurrent downloads
            range = f"bytes={offset}-{offset+length-1}"
            log.debug(f"reading {range} from {self.key.split('/')[-1]}")
            response = await self.s3.get_object(
                Bucket=self.bucket, Key=self.key, Range=range
            )
            return await response["Body"].read()

    def _find_data_offset(self, ext=1) -> int:
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

    def _find_pixel_offset(self, col, row) -> int:
        """Returns the byte offset of a specific pixel position."""
        pixel_offset = col + row * FFI_COLUMNS
        return self.data_offset + BYTES_PER_PIX * pixel_offset

    def _find_pixel_blocks(self, col, row, shape=(1, 1)) -> list:
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

    def cutout_array(self, col, row, shape=(5, 5)) -> np.array:
        """Returns a 2D array of pixel values."""
        blocks = self._find_pixel_blocks(col=col, row=row, shape=shape)
        bytedata = self.read_blocks(blocks)
        data = []
        for b in bytedata:
            n_pixels = len(b) // BYTES_PER_PIX
            values = struct.unpack(">" + "f" * n_pixels, b)
            data.append(values)
        return np.array(data)

    async def async_cutout_array(self, col, row, shape=(5, 5)) -> np.array:
        blocks = self._find_pixel_blocks(col=col, row=row, shape=shape)
        bytedata = await asyncio.gather(
            *[self.async_read_block(offset=blk[0], length=blk[1]) for blk in blocks]
        )
        data = []
        for b in bytedata:
            n_pixels = len(b) // BYTES_PER_PIX
            values = struct.unpack(">" + "f" * n_pixels, b)
            data.append(values)
        return np.array(data)

    def cutout(self, col, row, shape=(5, 5)) -> "Cutout":
        """Returns a 2D array of pixel values."""
        flux = self.cutout_array(col=col, row=row, shape=shape)
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

    async def async_cutout(self, col, row, shape=(5, 5)) -> "Cutout":
        """Returns a 2D array of pixel values."""
        flux = await self.async_cutout_array(col=col, row=row, shape=shape)
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


def list_images(sector, camera, ccd):
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    uris = fs.glob(
        f"stpubdata/tess/public/ffi/s{SECTOR:04d}/*/*/{CAMERA}-{CCD}/**_ffic.fits"
    )
    return [TessImage(uri) for uri in uris]


def data_offset_lookup(camera=1, ccd=1):
    """Returns a lookup table mapping sector => data offset."""
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
