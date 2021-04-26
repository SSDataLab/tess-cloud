import asyncio
from collections import UserList
from typing import Union, Tuple

from astropy.time import Time
from pandas import DataFrame
import tqdm

from tess_locator import TessCoord, TessCoordList

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

    @property
    def time(self):
        return [img.time for img in self]

    @property
    def quality(self):
        return [img.quality for img in self]

    @property
    def cadenceno(self):
        return [img.cadenceno for img in self]

    def to_pandas(self) -> DataFrame:
        data = [
            {
                "url": im.url,
                "sector": im.sector,
                "camera": im.camera,
                "ccd": im.ccd,
                "time": im.time,
                "cadenceno": im.cadenceno,
                "quality": im.quality,
            }
            for im in self
        ]
        return DataFrame(data)

    @classmethod
    def from_catalog(cls, catalog: DataFrame):
        series = catalog.apply(
            lambda x: TessImage(
                url=x["path"],
                data_offset=x["data_offset"],
                meta=x.to_dict(),
            ),
            axis=1,
        )
        obj = cls(series.values)
        obj._catalog = catalog
        return obj

    def _get_default_client(self):
        for img in self:
            if img.url:
                return img._get_default_client()
        # If all images are None, the client type doesn't matter
        return img._get_default_client(client_type="s3")

    async def _get_cutouts(self, crdlist: TessCoordList, shape: Tuple[int, int]):
        if len(self) == 0:
            return []
        async with self._get_default_client() as client:
            # Create list of functions to be executed
            flist = [
                img.async_cutout(
                    column=crd.column, row=crd.row, shape=shape, client=client
                )
                for img, crd in zip(self, crdlist)
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

    def cutout(self, column: int, row: int, shape: Tuple[int, int] = (5, 5)):
        # Turn (column, row) into a TessCoordList to match the interface of _get_cutouts
        crdlist = TessCoordList(
            [
                TessCoord(
                    sector=img.sector if img.url else 1,  # TessCoord needs valid number
                    camera=img.camera if img.url else 1,
                    ccd=img.ccd if img.url else 1,
                    column=column,
                    row=row,
                )
                for img in self
            ]
        )
        return self.moving_cutout(crdlist=crdlist, shape=shape)

    def moving_cutout(self, crdlist: TessCoordList, shape: Tuple[int, int] = (5, 5)):
        cutouts = asyncio.run(self._get_cutouts(crdlist=crdlist, shape=shape))
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

        return tica.list_tica_images(
            sector=sector, camera=camera, ccd=ccd, time=time, provider=provider
        )
    else:
        from . import spoc

        return spoc.list_spoc_images(
            sector=sector, camera=camera, ccd=ccd, time=time, provider=provider
        )
