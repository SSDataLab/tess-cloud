"""
A TESS TPF contains the following data:

ColDefs(
    name = 'TIME'; format = 'D'; unit = 'BJD - 2457000, days'; disp = 'D14.7'
    name = 'TIMECORR'; format = 'E'; unit = 'd'; disp = 'E14.7'
    name = 'CADENCENO'; format = 'J'; disp = 'I10'
    name = 'RAW_CNTS'; format = '143J'; unit = 'count'; null = -1; disp = 'I8'; dim = '(11,13)'
    name = 'FLUX'; format = '143E'; unit = 'e-/s'; disp = 'E14.7'; dim = '(11,13)'
    name = 'FLUX_ERR'; format = '143E'; unit = 'e-/s'; disp = 'E14.7'; dim = '(11,13)'
    name = 'FLUX_BKG'; format = '143E'; unit = 'e-/s'; disp = 'E14.7'; dim = '(11,13)'
    name = 'FLUX_BKG_ERR'; format = '143E'; unit = 'e-/s'; disp = 'E14.7'; dim = '(11,13)'
    name = 'QUALITY'; format = 'J'; disp = 'B16.16'
    name = 'POS_CORR1'; format = 'E'; unit = 'pixel'; disp = 'E14.7'
    name = 'POS_CORR2'; format = 'E'; unit = 'pixel'; disp = 'E14.7'
)
"""
from datetime import datetime
from typing import Optional

import numpy as np
from numpy import array, ndarray

from astropy.wcs import WCS
from astropy.io import fits
import lightkurve as lk


TPF_OPTIONAL_COLUMNS = {
    "CADENCENO": {"format": "J"},
    "QUALITY": {"format": "J"},
    "POS_CORR1": {"format": "E", "unit": "pixels"},
    "POS_CORR2": {"format": "E", "unit": "pixels"},
    "SECTOR": {"format": "I"},
    "CAMERA": {"format": "B"},
    "CCD": {"format": "B"},
    "CORNER_COLUMN": {"format": "I", "unit": "pixels"},
    "CORNER_ROW": {"format": "I", "unit": "pixels"},
    "TARGET_COLUMN": {"format": "E", "unit": "pixels"},
    "TARGET_ROW": {"format": "E", "unit": "pixels"},
    "URL": {"format": "A150"},
}


class TargetPixelFile:
    """Class representation of a Target Pixel File (TPF)."""

    def __init__(
        self,
        time: ndarray,
        flux: ndarray,
        flux_err: ndarray,
        flux_bkg: Optional[ndarray] = None,
        flux_bkg_err: Optional[ndarray] = None,
        wcs: Optional[WCS] = None,
        meta: Optional[dict] = None,
    ):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.flux_bkg = flux_bkg
        if self.flux_bkg is None:
            self.flux_bkg = np.empty(shape=flux.shape)
            self.flux_bkg[:] = np.nan
        self.flux_bkg_err = flux_bkg_err
        if self.flux_bkg_err is None:
            self.flux_bkg_err = np.empty(shape=flux.shape)
            self.flux_bkg_err[:] = np.nan
        self.wcs = wcs

        meta_default = {
            "TELESCOP": "TESS",
            "ORIGIN": "tess_cloud",
            "CREATOR": "tess_cloud.targetpixelfile",
            "DATE": datetime.now().strftime("%Y-%m-%d"),
        }
        if meta:
            meta_default.update(meta)
        self.meta = meta_default

        self._optional_column_data = {}

    def add_column(self, name, array, colspec=None):
        self._optional_column_data[name] = array
        if colspec:
            TPF_OPTIONAL_COLUMNS[name] = colspec

    @property
    def n_cadences(self):
        return self.time.shape[0]

    @property
    def n_columns(self):
        return self.flux.shape[1]

    @property
    def n_rows(self):
        return self.flux.shape[2]

    @property
    def timecorr(self):
        if not hasattr(self, "_timecorr"):
            self._timecorr = np.zeros(self.n_cadences, dtype="float32")
        return self._timecorr

    @property
    def raw_cnts(self):
        if not hasattr(self, "_raw_cnts"):
            self._raw_cnts = np.empty(
                (self.n_cadences, self.n_rows, self.n_columns), dtype="int"
            )
            self._raw_cnts[:, :, :] = -1
        return self._raw_cnts

    @property
    def pos_corr1(self):
        if not hasattr(self, "_pos_corr1"):
            self._pos_corr1 = np.zeros(self.n_cadences, dtype="float32")
        return self._pos_corr1

    @property
    def pos_corr2(self):
        if not hasattr(self, "_pos_corr2"):
            self._pos_corr2 = np.zeros(self.n_cadences, dtype="float32")
        return self._pos_corr2

    @staticmethod
    def from_cutouts(images: list) -> "TargetPixelFile":
        if len(images) > 0:
            shape = (len(images), images[0].flux.shape[0], images[0].flux.shape[1])
        else:
            shape = (0, 0, 0)
        flux = np.empty(shape)
        flux_err = np.empty(shape)
        for idx, img in enumerate(images):
            flux[idx] = img.flux
            flux_err[idx] = np.nan

        time = np.array([img.time for img in images])
        tpf = TargetPixelFile(
            time=time,
            flux=flux,
            flux_err=flux_err,
        )

        tpf.add_column(
            name="CADENCENO", array=np.array([img.cadenceno for img in images])
        )
        tpf.add_column(name="QUALITY", array=np.array([img.quality for img in images]))
        tpf.add_column(name="SECTOR", array=np.array([img.sector for img in images]))
        tpf.add_column(name="CAMERA", array=np.array([img.camera for img in images]))
        tpf.add_column(name="CCD", array=np.array([img.ccd for img in images]))
        tpf.add_column(
            name="CORNER_COLUMN", array=np.array([img.corner_column for img in images])
        )
        tpf.add_column(
            name="CORNER_ROW", array=np.array([img.corner_row for img in images])
        )
        tpf.add_column(
            name="TARGET_COLUMN", array=np.array([img.target_column for img in images])
        )
        tpf.add_column(
            name="TARGET_ROW", array=np.array([img.target_row for img in images])
        )
        tpf.add_column(name="URL", array=np.array([img.url for img in images]))
        return tpf

    @staticmethod
    def read(path) -> "TargetPixelFile":
        f = fits.open(path)
        tpf = TargetPixelFile(
            time=f[1].data["TIME"],
            cadenceno=f[1].data["CADENCENO"],
            flux=f[1].data["FLUX"],
            flux_err=f[1].data["FLUX_ERR"],
            quality=f[1].data["QUALITY"],
        )
        return tpf

    def write(self, *args, **kwargs):
        hdulist = self._create_hdulist()
        return hdulist.writeto(*args, **kwargs)

    def to_lightkurve(self, quality_bitmask=0):
        return lk.TessTargetPixelFile(
            self._create_hdulist(), quality_bitmask=quality_bitmask
        )

    def _create_hdulist(self):
        """Returns an astropy.io.fits.HDUList object."""
        return fits.HDUList(
            [
                self._create_primary_hdu(),
                self._create_table_extension(),
                self._create_aperture_extension(),
            ]
        )

    def _create_primary_hdu(self):
        """Returns the primary extension (#0)."""
        hdu = fits.PrimaryHDU()
        hdu.header.update(self.meta)
        return hdu

    @property
    def _coldim(self):
        return "({},{})".format(self.n_columns, self.n_rows)

    @property
    def _eformat(self):
        return "{}E".format(self.n_rows * self.n_columns)

    def _create_table_extension(self):
        """Create the 'TARGETTABLES' extension (i.e. extension #1)."""
        # Turn the data arrays into fits columns and initialize the HDU
        jformat = "{}J".format(self.n_rows * self.n_columns)
        cols = []
        cols.append(
            fits.Column(name="TIME", format="D", unit="BJD - 2457000", array=self.time)
        )
        cols.append(
            fits.Column(name="TIMECORR", format="E", unit="D", array=self.timecorr)
        )
        cols.append(
            fits.Column(
                name="RAW_CNTS",
                format=jformat,
                unit="count",
                dim=self._coldim,
                array=self.raw_cnts,
            )
        )
        cols.append(
            fits.Column(
                name="FLUX",
                format=self._eformat,
                unit="e-/s",
                dim=self._coldim,
                array=self.flux,
            )
        )
        cols.append(
            fits.Column(
                name="FLUX_ERR",
                format=self._eformat,
                unit="e-/s",
                dim=self._coldim,
                array=self.flux_err,
            )
        )
        cols.append(
            fits.Column(
                name="FLUX_BKG",
                format=self._eformat,
                unit="e-/s",
                dim=self._coldim,
                array=self.flux_bkg,
            )
        )
        cols.append(
            fits.Column(
                name="FLUX_BKG_ERR",
                format=self._eformat,
                unit="e-/s",
                dim=self._coldim,
                array=self.flux_bkg_err,
            )
        )

        for name in self._optional_column_data:
            key = name.upper()
            cols.append(
                fits.Column(
                    name=key,
                    array=self._optional_column_data[key],
                    format=TPF_OPTIONAL_COLUMNS[key].get("format", ""),
                    unit=TPF_OPTIONAL_COLUMNS[key].get("unit", ""),
                    dim=TPF_OPTIONAL_COLUMNS[key].get("dim", ""),
                )
            )

        coldefs = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.header["BJDREFI"] = 2457000

        return hdu

    def _create_aperture_extension(self):
        """Create the aperture mask extension (i.e. extension #2)."""
        mask = 3 * np.ones((self.n_rows, self.n_columns), dtype="int32")
        hdu = fits.ImageHDU(mask)
        hdu.header["EXTNAME"] = "APERTURE"
        return hdu
