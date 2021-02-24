from astropy.io import fits
import numpy as np

from tess_cloud import TessImage


def test_cutout():
    """Test basic features of RemoteTessImage."""
    url = "s3://stpubdata/tess/public/ffi/s0012/2019/142/2-1/tess2019142115932-s0012-2-1-0144-s_ffic.fits"
    img = TessImage(url=url)
    # Can we retrieve the first FITS header keyword? (SIMPLE)
    assert img.read_block(0, 6).decode("ascii") == "SIMPLE"
    assert img.read_blocks([(0, 6), (8, 1)]) == [b"SIMPLE", b"="]
    # Can we find the correct start position of the data for extenions 0 and 1?
    assert img._find_data_offset(ext=0) == 2880
    assert img._find_data_offset(ext=1) == 23040
    assert img._find_pixel_offset(col=0, row=0) == 23040
    assert img._find_pixel_blocks(col=0, row=0, shape=(1, 1)) == [(23040, 4)]
    # Corner pixel
    assert img.cutout_array(col=0, row=0, shape=(1, 1)).round(7) == 0.0941298
    # First three pixels of the first row
    assert (
        img.cutout_array(col=1, row=0, shape=(3, 1)).round(7)
        == np.array([0.0941298, -0.0605419, 0.0106343])
    ).all()
    # First three pixels of the first column
    assert (
        img.cutout_array(col=1, row=1, shape=(1, 3)).round(7)
        == np.array([[-0.0605419], [0.0327947], [-0.0278026]])
    ).all()
