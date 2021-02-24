"""
TODO
----
* TessImage should accept any file-like object in addition to a URI string.
"""
import re
import struct

import s3fs
import numpy as np

# FITS standard specifies that header and data units
# shall be a multiple of 2880 bytes long.
FITS_BLOCK_SIZE = 2880  # bytes

# TESS FFI dimensions
FFI_COLUMNS = 2136  # i.e. NAXIS1
FFI_ROWS = 2078  # i.e. NAXIS2

BYTES_PER_PIX = 4  # float32

S3FILESYSTEM = s3fs.S3FileSystem(anon=True)
# Use a small block size when reading from S3 to avoid wasting time on excessive buffering
S3_BLOCK_SIZE = 2880  # bytes


class TessImage:

    def __init__(self, url):
        self.url = url

    @property
    def _fileobj(self):
        if not hasattr(self, "__fileobj"):
            self.__fileobj = S3FILESYSTEM.open(self.url, block_size=S3_BLOCK_SIZE)
        return self.__fileobj

    @property
    def data_offset(self):
        if not hasattr(self, "_data_offset"):
            self._data_offset = self._find_data_offset(ext=1)
        return self._data_offset

    def read_header(self):
        pass

    def read_block(self, offset: int, length: int) -> bytes:
        """Returns a block of bytes from the file.

        Parameters
        ----------
        offset: int
            Byte offset to start read
        length: int
            Number of bytes to read
        """
        f = self._fileobj
        f.seek(offset, whence=0)
        return f.read(length)

    def read_blocks(self, blocks: list) -> bytes:
        result = []
        for blk in blocks:
            result.append(self.read_block(offset=blk[0], length=blk[1]))
        return result

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
    fs = s3fs.S3FileSystem(anon=True)
    uris = fs.glob(f"stpubdata/tess/public/ffi/s{SECTOR:04d}/*/*/{CAMERA}-{CCD}/**_ffic.fits")
    return [TessImage(uri) for uri in uris]
