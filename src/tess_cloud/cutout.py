import warnings

from tess_locator import locate, TessCoordList
from tess_ephem import ephem

from .image import TessImage
from .imagelist import list_images
from .targetpixelfile import TargetPixelFile


class TessCloudWarning(Warning):
    """Class for all tess-cloud warnings."""

    pass


def cutout_ffi(url, column, row, shape=(5, 5)) -> TargetPixelFile:
    """Retrieve a section from an FFI."""
    img = TessImage(url)
    cutout = img.cutout(column=column, row=row, shape=shape)
    return TargetPixelFile.from_cutouts([cutout]).to_lightkurve()


def cutout(
    target: str,
    shape: tuple = (5, 5),
    sector: int = None,
    author: str = "spoc",
    images: int = None,
) -> TargetPixelFile:
    """Returns a target pixel file."""
    locresult = locate(target=target, sector=sector)
    if len(locresult) < 1:
        raise ValueError("Target not observed by TESS.")
    if len(locresult) > 1:
        warnings.warn(
            f"Target observed in {len(locresult)} sector(s): "
            f"{', '.join([str(r.sector) for r in locresult])}. \n"
            f"Defaulting to sector {locresult[0].sector}. "
            f"You can change this by passing the `sector` argument.",
            TessCloudWarning,
        )
    crd = locresult[0]
    imagelist = crd.list_images(author=author)
    if images:
        imagelist = imagelist[:images]

    return imagelist.cutout(column=crd.column, row=crd.row, shape=shape)


def cutout_asteroid(
    target: str,
    shape: tuple = (10, 10),
    sector: int = None,
    author: str = "spoc",
    images: int = None,
) -> TargetPixelFile:
    """Returns a moving Target Pixel File centered on an asteroid."""
    eph_initial = ephem(target)

    if sector is None:
        all_sectors = eph_initial.sector.unique()
        # Default to first available sector
        sector = all_sectors[0]
        if len(all_sectors) > 1:
            warnings.warn(
                f"{target} has been observed in multiple sectors: {all_sectors}. "
                f"Defaulting to `sector={sector}`. "
                f"You can change the sector by passing the `sector` keyword argument. ",
                TessCloudWarning,
            )

    # Use most frequent (camera, ccd) combination to retrieve times
    camera, ccd = (
        eph_initial.query(f"sector == {sector}")
        .groupby(["camera", "ccd"])["ccd"]
        .count()
        .sort_values(ascending=False)
        .index[0]
    )
    # Get the exact mid-frame times
    time = list_images(sector=sector, camera=camera, ccd=ccd, author=author).time
    eph = ephem(target, time=time, verbose=True)
    if images:
        eph = eph[:images]

    crdlist = TessCoordList.from_pandas(eph)
    imagelist = crdlist.list_images(author=author)
    return imagelist.moving_cutout(crdlist=crdlist, shape=shape)
