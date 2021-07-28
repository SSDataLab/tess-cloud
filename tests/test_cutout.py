import asyncio

from astropy.io import fits
import lightkurve as lk
import numpy as np
import pytest

from tess_cloud import TessImage, list_images


def test_cutout():
    """Test basic features of RemoteTessImage."""
    url = "s3://stpubdata/tess/public/ffi/s0012/2019/142/2-1/tess2019142115932-s0012-2-1-0144-s_ffic.fits"
    img = TessImage(url=url)
    # Can we retrieve the first FITS header keyword? (SIMPLE)
    assert img.read_block(0, 6) == b"SIMPLE"
    assert img.read_block(8, 1) == b"="
    # assert img.read_blocks([(0, 6), (8, 1)]) == [b"SIMPLE", b"="]
    # Can we find the correct start position of the data for extenions 0 and 1?
    assert asyncio.run(img._find_data_offset(ext=0)) == 2880
    assert asyncio.run(img._find_data_offset(ext=1)) == 23040
    # By tess convention, the very first pixel has column=1 and row=1
    assert asyncio.run(img._find_pixel_offset(column=1, row=1)) == 23040
    assert asyncio.run(img._find_pixel_blocks(column=1, row=1, shape=(1, 1))) == [
        (23040, 4)
    ]
    # Corner pixel
    assert img.cutout(column=1, row=1, shape=(1, 1)).flux.round(7) == 0.0941298
    # First three pixels of the first row
    assert (
        img.cutout(column=2, row=1, shape=(3, 1)).flux.round(7)
        == np.array([0.0941298, -0.0605419, 0.0106343])
    ).all()
    # First three pixels of the first column
    assert (
        img.cutout(column=2, row=2, shape=(1, 3)).flux.round(7)
        == np.array([[-0.0605419], [0.0327947], [-0.0278026]])
    ).all()


@pytest.mark.remote_data
def test_against_tesscut():
    """Does a cutout from TessCut match TessCloud?"""
    target = "Pi Men"
    sector = 31
    shape = (3, 3)
    # Download TPF with TessCut
    tpf_tesscut = lk.search_tesscut(target, sector=sector).download(
        cutout_size=shape, quality_bitmask=None
    )
    # Download TPF with TessCloud
    imglist = list_images(sector=sector, camera=tpf_tesscut.camera, ccd=tpf_tesscut.ccd)
    center_column, center_row = (
        tpf_tesscut.column + 1,
        tpf_tesscut.row + 1,
    )  # add +1 to request center of a (3, 3)
    tpf_tesscloud = imglist[0:2].cutout(
        column=center_column, row=center_row, shape=shape
    )
    # Compare both
    assert np.all(tpf_tesscut[0].flux == tpf_tesscloud[0].flux)
    assert np.all(tpf_tesscut[1].flux == tpf_tesscloud[1].flux)
