from collections.abc import Mapping

import xarray as xr
import xbatcher as xb
from numba import guvectorize
from spatialdata.models import C, X, Y

from anatomize.core.decorators import convert_kwargs_to_xr_vec


def iter_channels(image: xr.DataArray) -> xb.BatchGenerator:  # noqa: D103
    n_c, n_y, n_x = image.shape
    return xb.BatchGenerator(image, input_dims={Y: n_y, X: n_x}, batch_dims={C: 1})


def _rechunk(image: xr.DataArray, chunks: Mapping[str, int] | None) -> xr.DataArray:
    _, n_y, n_x = image.shape
    if chunks is None:
        chunks = {C: 1, Y: n_y, X: n_x}
    return image.chunk(chunks)


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:,:])"],
    "(m,n),(c),(c)->(m,n)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _normalize(Ic, q_min, q_max, out):
    """Normalize an image.

    Parameters
    ----------
    Ic : NDArray
        The channel of the image to normalize.
    q_min : NDArray
        The minimum value of the channel.
    q_max : NDArray
        The maximum value of the channel.
    out : NDArray
        The normalized channel.
    """
    out[:] = (Ic[:] - q_min[:]) / (q_max[:] - q_min[:])


@xr.register_dataarray_accessor("an")
class AnatomizeDataArrayAccessor:
    """Accessor for DataArrays. Contains common preprocessing methods."""

    def __init__(self, obj: xr.DataArray):
        self._obj = obj


def normalize(
    image: xr.DataArray, q: tuple[float, float] = (0.0, 1.0), return_quantiles: bool = False, **quantile_kwargs
) -> xr.DataArray:
    """Normalize a DataArray's values between 0 and 1.

    Parameters
    ----------
    stretch : float, default 1.0
        A percentile stretch to apply before normalization between 0.0 and 0.1.

    Returns
    -------
    xarray.DataArray
        The dataset with normalized values.

    Raises
    ------
    ValueError
        If the stretch value is outside the valid range.
    """
    data = _rechunk(image, chunks=None)

    quantiles = data.quantile(q, dim=[Y, X], **quantile_kwargs)

    if return_quantiles:
        return quantiles
    else:
        q_min_xr, q_max_xr = quantiles

        return xr.apply_ufunc(
            _normalize,
            data,
            q_min_xr,
            q_max_xr,
            input_core_dims=[[Y, X], [], []],
            output_core_dims=[[Y, X]],
            dask="parallelized",
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:,:])"],
    "(m,n),(c),(c)->(m,n)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _gamma(Ic, gamma, gain, out):
    out[:] = (Ic[:] ** gamma[:]) * gain[:]


@convert_kwargs_to_xr_vec("gamma", "gain")
def adjust_gamma(image: xr.DataArray, gamma: float = 1, gain: float = 1) -> xr.DataArray:
    """Adjust the gamma of an image.

    Parameters
    ----------
    image : xarray.DataArray
        The image to adjust.
    gamma : float, default 1
        The gamma value to adjust the image by.
    gain : float, default 1
        The gain value to adjust the image by.

    Returns
    -------
    xarray.DataArray
        The adjusted image.
    """
    data = _rechunk(image, chunks=None)

    return xr.apply_ufunc(
        _gamma,
        data,
        gamma,
        gain,
        input_core_dims=[[Y, X], [], []],
        output_core_dims=[[Y, X]],
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
