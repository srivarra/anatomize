import xarray as xr
from skimage import exposure

import anatomize as an


def test_adjust_gamma():
    from spatialdata.datasets import blobs

    blobs_sdata = blobs()

    blobs_image_gamma_xr = an.exposure.adjust_gamma(image=blobs_sdata.images["blobs_image"], gamma=1.5, gain=2.0)

    blobs_image_gamma_skimage = exposure.adjust_gamma(blobs_sdata.images["blobs_image"], 1.5, 2.0)
    xr.testing.assert_equal(a=blobs_image_gamma_xr, b=blobs_image_gamma_skimage)
