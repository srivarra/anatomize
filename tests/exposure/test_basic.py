import pytest
import spatialdata as sd
import xarray as xr
from skimage import exposure
from upath import UPath

import anatomize.exposure as an_exposure


def test_adjust_gamma(blobs_sdata_store: UPath):
    blobs_sdata = sd.read_zarr(blobs_sdata_store)

    blobs_image_gamma_xr = an_exposure.adjust_gamma(image=blobs_sdata.images["blobs_image"], gamma=1.5, gain=2.0)

    blobs_image_gamma_skimage = xr.DataArray(
        data=exposure.adjust_gamma(blobs_sdata.images["blobs_image"], 1.5, 2.0),
        coords=blobs_sdata.images["blobs_image"].coords,
        dims=blobs_sdata.images["blobs_image"].dims,
    )
    xr.testing.assert_allclose(a=blobs_image_gamma_xr, b=blobs_image_gamma_skimage, atol=1e-10)


@pytest.mark.parametrize("gain,inv", ([1.5, True], [2.0, False]))
def test_adjust_log(blobs_sdata_store: UPath, gain: float, inv: bool):
    blobs_sdata = sd.read_zarr(blobs_sdata_store)

    blobs_image_log_xr = an_exposure.adjust_log(image=blobs_sdata.images["blobs_image"], gain=gain, inv=inv)

    blobs_image_log_skimage = xr.DataArray(
        data=exposure.adjust_log(blobs_sdata.images["blobs_image"], gain=gain, inv=inv),
        coords=blobs_sdata.images["blobs_image"].coords,
        dims=blobs_sdata.images["blobs_image"].dims,
    )
    xr.testing.assert_allclose(a=blobs_image_log_xr, b=blobs_image_log_skimage, atol=1e-10)


@pytest.mark.parametrize("cutoff,gain,inv", ([0.5, 10, True], [2.0, 20, False]))
def test_adjust_sigmoid(blobs_sdata_store: UPath, cutoff: float, gain: float, inv: bool):
    blobs_sdata = sd.read_zarr(blobs_sdata_store)

    blobs_image_sigmoid_xr = an_exposure.adjust_sigmoid(
        image=blobs_sdata.images["blobs_image"], cutoff=cutoff, gain=gain, inv=inv
    )

    blobs_image_sigmoid_skimage = xr.DataArray(
        data=exposure.adjust_sigmoid(blobs_sdata.images["blobs_image"], cutoff=cutoff, gain=gain, inv=inv),
        coords=blobs_sdata.images["blobs_image"].coords,
        dims=blobs_sdata.images["blobs_image"].dims,
    )
    xr.testing.assert_allclose(a=blobs_image_sigmoid_xr, b=blobs_image_sigmoid_skimage, atol=1e-10)


@pytest.mark.parametrize("in_range", ["image", "dtype", (0, 1)])
@pytest.mark.parametrize("out_range", ["image", "dtype", (0, 1)])
def test_rescale_intensity(
    blobs_sdata_store: UPath, in_range: str | tuple[float, float], out_range: str | tuple[float, float]
):
    blobs_sdata = sd.read_zarr(blobs_sdata_store)

    blobs_image_rescaled_xr = an_exposure.rescale_intensity(
        image=blobs_sdata.images["blobs_image"], in_range="image", out_range="dtype"
    )

    blobs_image_sigmoid_skimage = xr.DataArray(
        data=exposure.rescale_intensity(blobs_sdata.images["blobs_image"], in_range="image", out_range="dtype"),
        coords=blobs_sdata.images["blobs_image"].coords,
        dims=blobs_sdata.images["blobs_image"].dims,
    )

    xr.testing.assert_allclose(a=blobs_image_rescaled_xr, b=blobs_image_sigmoid_skimage, atol=1e-10)
