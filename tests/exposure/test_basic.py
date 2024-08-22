import pytest  # noqa: F401
import spatialdata as sd
import xarray as xr

# from confest import blobs_sdata_store
from skimage import exposure
from upath import UPath

import anatomize as an


def test_adjust_gamma(blobs_sdata_store: UPath):
    blobs_sdata = sd.read_zarr(blobs_sdata_store)

    blobs_image_gamma_xr = an.exposure.adjust_gamma(image=blobs_sdata.images["blobs_image"], gamma=1.5, gain=2.0)

    blobs_image_gamma_skimage = exposure.adjust_gamma(blobs_sdata.images["blobs_image"], 1.5, 2.0)
    xr.testing.assert_equal(a=blobs_image_gamma_xr, b=blobs_image_gamma_skimage)


def test_adjust_log(blobs_sdata_store: UPath):
    blobs_sdata = sd.read_zarr(blobs_sdata_store)

    blobs_image_log_xr = an.exposure.adjust_log(image=blobs_sdata.images["blobs_image"], gain=2.0)

    blobs_image_log_skimage = exposure.adjust_log(blobs_sdata.images["blobs_image"], 2.0)
    xr.testing.assert_equal(a=blobs_image_log_xr, b=blobs_image_log_skimage)
