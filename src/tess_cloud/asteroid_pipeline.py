"""Implements a simple pipeline to create moving Target Pixel Files
to which a simple median background correction is applied."""
from typing import List, Tuple
import warnings

import numpy as np

from tess_ephem import ephem

from . import cutout_asteroid, __version__
from .targetpixelfile import TargetPixelFile, TPF_OPTIONAL_COLUMNS


class SimpleAsteroidPipeline:
    """Simple TESS Asteroid Data Reduction pipeline.

    This class created a moving Target Pixel File and enables a simple median
    background model to be subtracted.
    """

    target_tpf = None
    background_tpfs = None

    def __init__(
        self,
        target="Juno",
        shape=(10, 10),
        sector=23,
        author="SPOC",
        provider=None,
        images=None,
    ):
        self.target = target
        self.shape = shape
        self.sector = sector
        self.author = author
        self.provider = provider
        self.images = images

    def _minimum_time_delay(self):
        """Returns the minimum amount of time the target takes
        to move out of the aperture. In units of days."""
        # Retrieve object ephemeris:
        eph = ephem(self.target, sector=self.sector, verbose=True)
        # Compute the diagonal aperture size in pixels:
        diagonal = 1.414 * max(self.shape)
        # Minimum time needed for the target to move across the diagonal:
        diagonal_crossing_time = diagonal / eph["pixels_per_hour"].min()
        # Time needed to move across half the diagonal in units of days:
        result = 0.5 * diagonal_crossing_time / 24
        return result

    def compute_delays(self):
        """Returns the default time deltas for the leading/lagging apertures."""
        buffer = 30.0 / 1440.0  # Add 30 minutes buffer to be safe
        min_delta = round(buffer + self._minimum_time_delay(), 2)
        return [x * min_delta for x in [-1, +1, +2]]

    def _cutout_target(self):
        return self._cutout_background_tpfs(delays=[0.0])[0]

    def _cutout_background_tpfs(self, delays: List[float]):
        background_tpfs = []
        for delay in delays:
            tpf = cutout_asteroid(
                target=self.target,
                shape=self.shape,
                sector=self.sector,
                author=self.author,
                provider=self.provider,
                images=self.images,
                time_delay=delay,
            )
            background_tpfs.append(tpf)
        return background_tpfs

    def estimate_background(
        self, delays: Tuple[float] = (-1, +1)
    ) -> Tuple[float, float]:
        self.background_tpfs = self._cutout_background_tpfs(delays=delays)
        median = np.nanmedian([tpf.flux for tpf in self.background_tpfs], axis=0)
        std = np.nanstd([tpf.flux for tpf in self.background_tpfs], axis=0)
        return (median, std)

    def run(self, delays: Tuple[float] = None):
        if not delays:
            delays = self.compute_delays()

        self.target_tpf = self._cutout_target()
        self.flux_bkg, self.flux_bkg_err = self.estimate_background(delays=delays)

        corrected_flux = self.target_tpf.flux.value - self.flux_bkg

        # Flux values can accidentally be negative
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corrected_flux_err = np.sqrt(
                self.target_tpf.flux.value + self.flux_bkg_err ** 2
            )

        meta = {
            "ORIGIN": f"tess_cloud v{__version__}",
            "CREATOR": "tess_cloud.targetpixelfile",
            "METHOD": "SimpleAsteroidPipeline",
            "TARGET": (self.target, "Moving target identifier"),
            "SECTOR": (self.sector, "TESS sector number"),
            "FFI_AUTH": (self.author, "Author of the FFI images used"),
            "FFI_PROV": (self.provider, "Data server accessed"),
            "START": (
                self.target_tpf.time[0].value,
                "Time of the first cadence [BTJD]",
            ),
            "STOP": (self.target_tpf.time[-1].value, "Time of the last cadence [BTJD]"),
        }
        for idx, delay in enumerate(delays):
            meta[f"BGDELAY{idx}"] = (
                delays[idx],
                f"Background aperture {idx} delay [days]",
            )

        tpf = TargetPixelFile(
            time=self.target_tpf.time.value,
            flux=corrected_flux,
            flux_err=corrected_flux_err,
            flux_bkg=self.flux_bkg,
            flux_bkg_err=self.flux_bkg_err,
            meta=meta,
        )
        tpf.add_column("QUALITY", self.target_tpf.quality)
        tpf.add_column("CADENCENO", self.target_tpf.cadenceno)
        tpf.add_column(
            "FLUX_ORIG",
            self.target_tpf.flux.value,
            colspec={"format": tpf._eformat, "dim": tpf._coldim, "unit": "e-/s"},
        )
        for idx, bkgtpf in enumerate(self.background_tpfs):
            tpf.add_column(
                f"FLUX_BKG_{idx}",
                bkgtpf.flux.value,
                colspec={"format": tpf._eformat, "dim": tpf._coldim, "unit": "e-/s"},
            )

        for col in [
            "SECTOR",
            "CAMERA",
            "CCD",
            "CORNER_COLUMN",
            "CORNER_ROW",
            "TARGET_COLUMN",
            "TARGET_ROW",
            "URL",
        ]:
            tpf.add_column(col, self.target_tpf.hdu[1].data[col])

        return tpf.to_lightkurve()
