import warnings
from collections.abc import Mapping, Sequence
from typing import Literal, Unpack

import datatree as dt
import numpy as np
import xarray as xr
import xbatcher as xb
from numpydantic import NDArray
from xarray.core.types import QuantileMethods

from anatomize.core import AxisSpec, C, X, Y, convert_kwargs_to_xr_vec, rechunk_image
from anatomize.core._typeddict_kwargs import QuantileKwargs

from ._utils import (
    _gamma,
    _histogram,
    _intensity_range,
    _inv_log,
    _inv_sigmoid,
    _log,
    _normalize,
    _output_dtype,
    _sigmoid,
)


def iter_channels(image: xr.DataArray) -> xb.BatchGenerator:  # noqa: D103
    n_c, n_y, n_x = image.shape
    return xb.BatchGenerator(image, input_dims={Y: n_y, X: n_x}, batch_dims={C: 1})


@rechunk_image
def normalize(
    image: xr.DataArray | dt.DataTree,
    q: tuple[float, float] = (0.0, 1.0),
    method: QuantileMethods = "linear",
    return_quantiles: bool = False,
    chunk: Mapping[str, int] | None = None,
    **quantile_kwargs: Unpack[QuantileKwargs],
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
    quantiles = image.quantile(q=q, dim=[Y, X], method=method, **quantile_kwargs)
    if return_quantiles:
        return quantiles
    else:
        q_min_xr, q_max_xr = quantiles

        return xr.apply_ufunc(
            _normalize,
            image,
            q_min_xr,
            q_max_xr,
            input_core_dims=[[Y, X], [], []],
            output_core_dims=[[Y, X]],
            dask="parallelized",
            output_dtypes=[image.dtype],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )


@rechunk_image
@convert_kwargs_to_xr_vec("gamma", "gain")
def adjust_gamma(
    image: xr.DataArray, gamma: float = 1, gain: float = 1, chunks: Mapping[str, int] | None = None
) -> xr.DataArray:
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
    return xr.apply_ufunc(
        _gamma,
        image,
        gamma,
        gain,
        input_core_dims=[[Y, X], [], []],
        output_core_dims=[[Y, X]],
        dask="parallelized",
        output_dtypes=[image.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


@rechunk_image
@convert_kwargs_to_xr_vec("gain")
def adjust_log(
    image: xr.DataArray, gain: float = 1, inv=False, chunks: Mapping[str, int] | None = None
) -> xr.DataArray:
    """Transforms the image pixelwise using the logarithmic or inverse logarithmic correction.

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

    f = _log if not inv else _inv_log

    return xr.apply_ufunc(
        f,
        image,
        gain,
        input_core_dims=[[Y, X], []],
        output_core_dims=[[Y, X]],
        dask="parallelized",
        output_dtypes=[image.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


@rechunk_image
@convert_kwargs_to_xr_vec("cutoff", "gain")
def adjust_sigmoid(
    image: xr.DataArray, cutoff: float = 0.5, gain: float = 10, inv=False, chunks: Mapping[str, int] | None = None
) -> xr.DataArray:
    """Performs Sigmoid Correction on the input image.

    Parameters
    ----------
    image
        The image to adjust.
    cutoff
        Cutoff of the sigmoid function that shifts the characteristic curve in horizontal direction, by default 0.5
    gain
        The constant multiplier in exponential's power of sigmoid function, by default 10
    inv
        If True, the negative sigmoid correction is performed, otherwise
        the sigmoid correction is performed, by default False.

    Returns
    -------
    The sigmoid adjusted image.
    """
    f = _sigmoid if not inv else _inv_sigmoid

    return xr.apply_ufunc(
        f,
        image,
        cutoff,
        gain,
        input_core_dims=[[Y, X], [], []],
        output_core_dims=[[Y, X]],
        dask="parallelized",
        output_dtypes=[image.dtype],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


@rechunk_image
def histogram(
    image: xr.DataArray,
    bins: Sequence[AxisSpec] | Mapping[str, AxisSpec] | AxisSpec,
    weight: xr.DataArray | None = None,
    density: bool = False,
    chunks: Mapping[str, int] | None = None,
) -> xr.DataArray:
    """Compute the histogram of the multichannel image.

    Parameters
    ----------
    image
        The image to compute the histogram of.
    bins
        Sequence of specifications for the histogram bins, in the same order as the
        variables of `data`. These are passed to :func:`xarray-histogram.histogram`

        Specification can either be:

        * a :class:`boost_histogram.axis.Axis`.
        * a tuple consisting of (number of bins, minimum value, maximum value) in which case the
          bins will be linearly spaced
        * only the number of bins, the minimum and maximum values are then computed from
          the data on the spot.
        * A mapping of channel names `C` to the specifications for the histogram bins.
    density
        If true normalize the histogram so that its integral is one. Does not take into account weight, by default False.

    Returns
    -------
    DataArray named `image_histogram`. The bins coordinates are named image_bins.
    """
    match bins:
        case int():
            bins = (bins,)
        case _:
            pass

    hist = _histogram(image, bins=bins, dims=[Y, X], density=density)
    return hist


@rechunk_image
def cumulative_distribution(
    image: xr.DataArray,
    bins: Sequence[AxisSpec] | Mapping[str, AxisSpec] | AxisSpec,
    density: bool = False,
    chunks: Mapping[str, int] | None = None,
) -> xr.DataArray:
    """Compute the cumulative distribution of the image.

    Parameters
    ----------
    image
        The image to compute the cumulative distribution of.
    bins
        The number of bins to use for the cumulative distribution, by default 256.

    Returns
    -------
    xr.DataArray
        The cumulative distribution of the image.
    """
    data_hist = histogram(image, bins, density=density)  # noqa: F841

    # data_hist.hist.cdf(x=)

    # return xr.apply_ufunc(
    #     _cumulative_distribution,
    #     data,
    #     nbins,
    #     input_core_dims=[[Y, X], []],
    #     output_core_dims=[[Y, X]],
    #     dask="parallelized",
    #     output_dtypes=[data.dtype],
    #     dask_gufunc_kwargs={"allow_rechunk": True},
    # )


@rechunk_image
def equalize_histogram(
    image: xr.DataArray,
    bins: Sequence[AxisSpec] | Mapping[str, AxisSpec] | AxisSpec,
    density: bool = False,
    mask: xr.DataArray | NDArray | None = None,
    chunks: Mapping[str, int] | None = None,
):
    """#TODO."""
    ...


@rechunk_image
def equalize_clhae(
    image: xr.DataArray,
    bins: Sequence[AxisSpec] | Mapping[str, AxisSpec] | AxisSpec,
    density: bool = False,
    mask: xr.DataArray | NDArray | None = None,
    chunks: Mapping[str, int] | None = None,
):
    """#TODO."""
    ...


@rechunk_image
@convert_kwargs_to_xr_vec("fraction_threshold")
def is_low_contrast(
    image: xr.DataArray,
    q: tuple[float, float] = (0.1, 0.9),
    method: QuantileMethods = "linear",
    fraction_threshold: float = 0.05,
    chunks: Mapping[str, int] | None = None,
    **quantile_kwargs: Unpack[QuantileKwargs],
) -> xr.DataArray:
    """#TODO."""
    min_q, max_q = image.quantile(q=q, dim=[Y, X], method=method, **quantile_kwargs)

    ratio: xr.DataArray = max_q - min_q

    return ratio < fraction_threshold


def rescale_intensity(
    image: xr.DataArray,
    in_range: Literal["image", "dtype"] | tuple[int, int] = "image",
    out_range: Literal["image", "dtype"] | tuple[int, int] = "dtype",
):
    """#TODO."""
    if out_range in ["image", "dtype"]:
        out_dtype = _output_dtype(image.dtype.type, image.dtype)
    else:
        out_dtype = _output_dtype(out_range, image.dtype)

    in_min, in_max = map(float, _intensity_range(image, in_range))
    out_min, out_max = map(float, _intensity_range(image, out_range, clip_negative=(in_min >= 0)))
    if np.any(nans := np.isnan([in_min, in_max, out_min, out_max])):
        # Get the value which is nan from the nans list
        nan_variables = [
            var for var, is_nan in zip(["in_min", "in_max", "out_min", "out_max"], nans, strict=False) if is_nan
        ]
        warnings.warn(
            message=(
                f"One or more intensity levels are NaN: {', '.join(nan_variables)}. Rescaling will broadcast NaN to the full image."
                "To avoid this, provide the intensity levels yourself e.g. (`np.nanmin(image)`, `np.nanmax(image)`)"
            ),
            stacklevel=2,
        )
    image_input_clip = image.clip(min=in_min, max=in_max).compute()
    if in_min != in_max:
        image_output_clip = (
            (((image_input_clip - in_min) / (in_max - in_min)) * (out_max - out_min) + out_min)
            .compute()
            .astype(dtype=out_dtype)
        )
    else:
        image_output_clip = image_input_clip.clip(min=out_min, max=out_max).compute().astype(dtype=out_dtype)
    return image_output_clip
