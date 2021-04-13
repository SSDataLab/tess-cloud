import numpy as np

from tess_cloud.image import TessImage
from tess_cloud.imagelist import TessImageList


def test_missing_images():
    """Asteroid background correction relies on being able to have empty images,
    because a background frame may not be available at every timestamp."""
    img = TessImage(None)
    assert img.url is None
    assert np.isnan(img.time)
    assert np.isnan(img.sector)
    assert np.isnan(img.camera)
    assert np.isnan(img.ccd)

    # Can we make a cutout from a list of missing images?
    imglist = TessImageList([TessImage(None)] * 3)
    assert all(np.isnan(imglist.time))
    tpf = imglist.cutout(200, 200, shape=(4, 5))
    assert tpf.flux.shape == (3, 5, 4)
    assert np.isnan(tpf.flux).all()
    assert tpf.time.format == "btjd"
