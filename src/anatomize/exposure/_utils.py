from collections.abc import Collection, Hashable, Sequence
from typing import Literal, Unpack

import boost_histogram as bh
import dask_histogram as dh
import numpy as np
import xarray as xr
from numba import guvectorize
from numpydantic import NDArray, dtype
from numpydantic.validation import validate_dtype

from anatomize.core._typeddict_kwargs import PartitionedFactoryKwargs
from anatomize.core.models import AxisSpec


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


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:,:])"],
    "(m,n),(c),(c)->(m,n)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _gamma(Ic, gamma, gain, out):
    out[:] = (Ic[:] ** gamma[:]) * gain[:]


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:,:])"],
    "(m,n),(c)->(m,n)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _log(Ic, gain, out):
    out[:] = gain[:] * np.log2(1 + Ic[:])


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:,:])"],
    "(m,n),(c)->(m,n)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _inv_log(Ic, gain, out):
    out[:] = gain[:] * (2 ** Ic[:] - 1)


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:,:])"],
    "(m,n),(c),(c)->(m,n)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _sigmoid(Ic, cutoff, gain, out):
    out[:] = 1 / (1 + np.exp(gain[:] * (cutoff[:] - Ic[:])))


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:,:])"],
    "(m,n),(c),(c)->(m,n)",
    nopython=True,
    cache=True,
    target="cpu",
)
def _inv_sigmoid(Ic, cutoff, gain, out):
    out[:] = 1 - (1 / (1 + np.exp(gain[:] * (cutoff[:] - Ic[:]))))


VAR_WEIGHT = "_weight"
LOOP_DIM = "__loop_var"


def _histogram(
    *data: xr.DataArray,
    bins: Sequence[AxisSpec],
    dims: Collection[Hashable] | None = None,
    weight: xr.DataArray | None = None,
    density: bool = False,
    **partitioned_factory_kwargs: Unpack[PartitionedFactoryKwargs],
) -> xr.DataArray:
    """Compute histogram.

    Parameters
    ----------
    data
        The arrays to compute the histogram from. To compute a multi-dimensional
        histogram supply a sequence of as many arrays as the histogram dimensionality.
        Arrays must be broadcastable against each other. If any underlying data is a
        dask array, other inputs will be transformed into a dask array of a single
        chunk.
    bins
        Sequence of specifications for the histogram bins, in the same order as the
        variables of `data`.

        Specification can either be:

        * a :class:`boost_histogram.axis.Axis`.
        * a tuple of (number of bins, minimum value, maximum value) in which case the
          bins will be linearly spaced
        * only the number of bins, the minimum and maximum values are then computed from
          the data on the spot.
    dims
        Dimensions to compute the histogram along to. If left to None the
        data is flattened along all axis.
    weight
        Array of the weights to apply for each data-point.
    density
        If true normalize the histogram so that its integral is one.
        Does not take into account `weight`. Default is false.
    partitioned_factory_kwargs
        Passed to :func:`dask_histogram.partitioned_factory`.

    Returns
    -------
    histogram
        DataArray named ``<variables names>_histogram`` (for multi-dimensional
        histograms the names are separated by underscores). The bins coordinates are
        named ``<variable name>_bins``.
    """
    data_sanity_check(data)
    variables = [a.name for a in data]

    bins = manage_bins_input(bins, data)
    bins_names = [f"{v}_bins" for v in variables]

    if weight is not None:
        weight = weight.rename(VAR_WEIGHT)
        data = data + (weight,)

    data = xr.broadcast(*data)

    data = tuple(a.chunk({}) for a in data)
    data = xr.unify_chunks(*data)

    # Merge everything together so it can be sent through a single
    # groupby call.
    ds: xr.Dataset = xr.merge(data, join="exact")

    data_dims = ds.dims

    if dims is None:
        dims = data_dims

    # dimensions that we loop over
    dims_loop = set(data_dims) - set(dims)

    # dimensions that are chunked. We need to manually aggregate them
    dims_loop = set(data_dims) - set(dims)
    dims_aggr = {
        dim for var in variables for dim, sizes in ds[var].chunksizes.items() if any(s != ds.sizes[dim] for s in sizes)
    }
    dims_aggr -= dims_loop

    if len(dims_loop) == 0:
        # on flattened array
        hist = comp_hist_dask(ds, variables, bins, bins_names, **partitioned_factory_kwargs)
    else:
        stacked = ds.stack({LOOP_DIM: dims_loop})  # noqa: PD013

        hist = stacked.groupby(LOOP_DIM, squeeze=False).map(
            comp_hist_dask, shortcut=True, args=[variables, bins, bins_names], **partitioned_factory_kwargs
        )
        hist = hist.unstack()  # noqa: PD010

    for name, b in zip(bins_names, bins, strict=True):
        hist = hist.assign_coords({name: b.edges[:-1]})
        hist[name].attrs["right_edge"] = b.edges[-1]

    if density:
        widths = [np.diff(b.edges) for b in bins]
        if len(widths) == 1:
            area = widths[0]
        elif len(widths) == 2:
            area = np.outer(*widths)
        else:
            area = np.prod(np.array(np.ix_(*widths), dtype=object))

        area_xr = xr.DataArray(area, dims=bins_names)
        hist = hist / area_xr / hist.sum(bins_names)

    hist_name = "pdf" if density else "histogram"
    hist = hist.rename("_".join(map(str, variables + [hist_name])))
    return hist


def _separate_ravel(ds: xr.Dataset, variables: Sequence[Hashable]) -> tuple[NDArray, NDArray | None]:
    """Separate data and weight arrays and flatten arrays.

    Returns
    -------
    data
        List of data arrays.
    weight
        Array of weights if present in dataset, None otherwise.
    """
    data = [ds[v].data.ravel() for v in variables]
    weight = ds[VAR_WEIGHT].data.ravel() if VAR_WEIGHT in ds else None
    return data, weight


def comp_hist_dask(
    ds: xr.Dataset,
    variables: Sequence[Hashable],
    bins: Sequence[bh.axis.Axis],
    bins_names: Sequence[Hashable],
    **partitioned_factory_kwargs: Unpack[PartitionedFactoryKwargs],
) -> xr.DataArray:
    """Compute histogram for dask data."""
    data, weight = _separate_ravel(ds, variables)
    hist = dh.partitioned_factory(*data, axes=bins, weights=weight, **partitioned_factory_kwargs)  # type: ignore
    values, *_ = hist.collapse().to_dask_array()
    return xr.DataArray(data=values, dims=bins_names)


def data_sanity_check(data: Sequence[xr.DataArray]):
    """Ensure data is correctly formated.

    Raises
    ------
    TypeError
        If a 0-length sequence was supplied.
    TypeError
        If any data is not a :class:`xarray.DataArray`.
    ValueError
        If set of dimensions are not identical in all arrays.
    """
    if len(data) == 0:
        raise TypeError("Data sequence of length 0.")
    for a in data:
        if not isinstance(a, xr.DataArray):
            raise TypeError("Data must be a xr.DataArray, " f"a type {type(a).__name__} was supplied.")
    dims0 = set(data[0].dims)
    for a in data:
        if set(a.dims) != dims0:
            raise ValueError("Dimensions are different in supplied arrays.")


def weight_sanity_check(weight: Sequence[xr.DataArray], data: Sequence[xr.DataArray]):
    """Ensure weight is correctly formated.

    Raises
    ------
    TypeError
        If weight is not a :class:`xarray.DataArray`.
    ValueError
        If the set of dimensions are not the same in weights as data.
    """
    dims0 = set(data[0].dims)
    if not isinstance(weight, xr.DataArray):
        raise TypeError("Weights must be a xr.DataArray, " f"a type {type(weight).__name__} was supplied.")
    if set(weight.dims) != dims0:
        raise ValueError("Dimensions are different in supplied weights.")


def manage_bins_input(bins: Sequence[AxisSpec], data: Sequence[xr.DataArray]) -> Sequence[bh.axis.Axis]:
    """Check bins input and convert to boost objects.

    Raises
    ------
    ValueError
        If there are not as many bins specifications as data arrays.
    """
    if len(bins) != len(data):
        raise ValueError(f"Mismatch between number of bins ({len(bins)}) and data arrays ({len(data)}).")

    bins_out = []
    for spec, a in zip(bins, data):  # noqa: B905
        match spec:
            case bh.axis.Axis():
                bins_out.append(spec)
            case int():
                bins_out.append(bh.axis.Regular(spec, float(a.min()), float(a.max())))
            case tuple() if len(spec) == 3 and isinstance(spec[0], int):
                bins_out.append(bh.axis.Regular(spec[0], spec[1], spec[2]))
            case _:
                raise TypeError(f"Invalid bin specification: {spec}")
    return bins_out


new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    "g": np.float64,  # np.float128 ; doesn't exist on windows
    "G": np.complex128,  # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex: False):
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == "c":
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def _output_dtype(dtype_or_range, image_dtype):
    if isinstance(dtype_or_range, list | tuple | NDArray):
        # pair of values: always return float.
        return _supported_float_type(image_dtype, allow_complex=False)
    if isinstance(dtype_or_range, dtype.Dtype):
        # already a type: return it
        return dtype_or_range
    if validate_dtype(dtype=dtype_or_range, target=dtype.Float + dtype.Integer):
        try:
            # if it's a canonical numpy dtype, convert
            return np.dtype(dtype_or_range).type
        except TypeError:  # uint10, uint12, uint14
            # otherwise, return uint16
            return np.uint16
    else:
        raise ValueError(
            "Incorrect value for out_range, should be a valid image data "
            f"type or a pair of values, got {dtype_or_range}."
        )


dtype_range = {
    bool: (False, True),
    dtype.Bool: (False, True),
    **{d: (-1, 1) for d in dtype.Float},
}


def _intensity_range(
    image: xr.DataArray, range_values: Literal["image", "dtype"] | tuple[int, int], clip_negative: bool = False
):
    if range_values == "dtype":
        range_values = image.dtype.type

    if range_values == "image":
        i_min = image.min()
        i_max = image.max()
    elif range_values in dtype_range:
        i_min, i_max = dtype_range[range_values]
        if clip_negative:
            i_min = max(i_min, 0)
    elif np.all([validate_dtype(dtype=np.array(rv).dtype, target=dtype.Number) for rv in range_values]):
        i_min, i_max = (np.array(rv) if isinstance(rv, int | float) else rv for rv in range_values)
    else:
        raise ValueError(
            "Incorrect value for range_values, should be a valid input",
            f"type or a pair of values, got {range_values}.",
        )
    return (i_min, i_max)
