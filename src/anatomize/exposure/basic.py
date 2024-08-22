import xarray as xr
import xbatcher as xb
from spatialdata.models import C, X, Y

from anatomize.core.decorators import convert_kwargs_to_xr_vec
from anatomize.exposure._utils import _gamma, _inv_log, _log, _normalize, _rechunk


def iter_channels(image: xr.DataArray) -> xb.BatchGenerator:  # noqa: D103
    n_c, n_y, n_x = image.shape
    return xb.BatchGenerator(image, input_dims={Y: n_y, X: n_x}, batch_dims={C: 1})


def normalize(
    image: xr.DataArray, q: tuple[float, float] = (0.0, 1.0), return_quantiles: bool = False, **quantile_kwargs
) -> xr.DataArray:
    """Normalize a DataArray's values between 0 and 1 using the Quantile method.

    Parameters
    ----------
    image
        The image to normalize.
    q
        The minimum and maximum values to normalize the image to, by default (0.0, 1.0)
    return_quantiles
        Return the minimum and maximum values used to normalize the image, by default False.
        If False, the normalized image is returned, if True a DataArray with the quantiles is returned.
    **quantile_kwargs
        Additional keyword arguments to pass to `xr.DataArray.quantile`

    Returns
    -------
    xr.DataArray
        Either the normalized image or a DataArray with the quantiles used to normalize the image.
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


@convert_kwargs_to_xr_vec("gamma", "gain")
def adjust_gamma(image: xr.DataArray, gamma: float = 1, gain: float = 1) -> xr.DataArray:
    """Performs Gamma Correction on the input image.

    Parameters
    ----------
    image
        The image to adjust.
    gamma
        The gamma value to adjust the image by, by default 1
    gain
        The gain value to adjust the image by, by default 1

    Returns
    -------
    xr.DataArray
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


@convert_kwargs_to_xr_vec("gain")
def adjust_log(image: xr.DataArray, gain: float = 1, inv=False) -> xr.DataArray:
    r"""Transforms the image pixelwise using the logarithmic or inverse logarithmic correction.

    Parameters
    ----------
    image
        The image to adjust.
    gain
        The constant multiplier, by default 1.
    inv
        If True, the inverse logarithmic correction is performed, otherwise
        the logarithmic correction is performed, by default False.

    Returns
    -------
    The logarithmic or inverse logarithmic adjusted image.
    """
    ...

    data = _rechunk(image, chunks=None)

    f = _log if not inv else _inv_log

    return xr.apply_ufunc(
        f,
        data,
        gain,
        input_core_dims=[[Y, X], []],
        output_core_dims=[[Y, X]],
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
