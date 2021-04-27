import warnings
from typing import Optional

from tess_locator import locate, TessCoordList
from tess_ephem import ephem

from astropy.time import Time

from .image import TessImage
from .imagelist import list_images, TessImageList
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
    sector: Optional[int] = None,
    author: str = "spoc",
    provider: Optional[str] = None,
    images: Optional[int] = None,
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
    imagelist = crd.list_images(author=author, provider=provider)
    if images:
        imagelist = imagelist[:images]

    return imagelist.cutout(column=crd.column, row=crd.row, shape=shape)


def cutout_asteroid(
    target: str,
    shape: tuple = (10, 10),
    sector: Optional[int] = None,
    author: str = "spoc",
    provider: Optional[str] = None,
    images: Optional[int] = None,
    time_delay: float = 0.0,
) -> TargetPixelFile:
    """Returns a moving Target Pixel File centered on an asteroid."""
    eph_initial = ephem(target, sector=sector)

    # If no sector is specified, we default to the first available sector
    # and issue a warning if multiple sector choices were available.
    if sector is None:
        all_sectors = eph_initial.sector.unique()
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
    time = list_images(
        sector=sector, camera=camera, ccd=ccd, author=author, provider=provider
    ).time
    eph = ephem(target, time=time, verbose=True)
    # If a time delay is requested, change the times
    if time_delay != 0:
        eph.index -= time_delay

    if images:
        eph = eph[:images]

    crdlist = TessCoordList.from_pandas(eph)
    # crdlist = TessCoordList([crd for crd in crdlist if crd.is_observed()])
    # imagelist = crdlist.list_images(author=author)

    # Make sure a time-delay cutout has the same number of frames
    imagelist = TessImageList([])
    for crd in crdlist:
        if crd.is_observed():
            imagelist.append(crd.list_images(author=author, provider=provider)[0])
        else:
            imagelist.append(TessImage(None))

    return imagelist.moving_cutout(crdlist=crdlist, shape=shape)
