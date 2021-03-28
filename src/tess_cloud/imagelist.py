import asyncio
from collections import UserList
from typing import Union

from astropy.time import Time
from pandas import DataFrame
import tqdm

from .image import TessImage
from .targetpixelfile import TargetPixelFile


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
            lambda x: TessImage(url=x[0], data_offset=x[7]), axis=1, raw=True
        )
        return cls(series.values)

    async def _get_cutouts(self, column, row, shape):
        async with self[0]._get_default_client() as client:
            # Create list of functions to be executed
            flist = [
                img.async_cutout(column=column, row=row, shape=shape, client=client)
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

    def cutout(self, column: int, row: int, shape=(5, 5)):
        cutouts = asyncio.run(self._get_cutouts(column=column, row=row, shape=shape))
        tpf = TargetPixelFile.from_cutouts(cutouts)
        return tpf.to_lightkurve()


def list_images(
    sector: int = 1,
    camera: int = 1,
    ccd: int = 1,
    time: Union[str, Time] = None,
    author: str = "spoc",
    provider: str = None,
) -> TessImageList:
    """Returns the list of FFI images."""
    if author == "tica":
        from . import tica

        return tica.list_tica_images(sector=sector, camera=camera, ccd=ccd)
    else:
        from . import spoc

        return spoc.list_spoc_images(
            sector=sector, camera=camera, ccd=ccd, time=time, provider=provider
        )
