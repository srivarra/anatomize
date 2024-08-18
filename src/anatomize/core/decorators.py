import inspect
from collections.abc import Callable
from functools import wraps
from inspect import BoundArguments
from typing import Concatenate, ParamSpec, TypeVar

import dask.array as da
import numpy as np
import xarray as xr
from spatialdata.models import C, get_channels

P = ParamSpec("P")
T = TypeVar("T")


def _kwarg_to_vec(value, n_c, cs, name):
    return xr.DataArray(data=da.from_array(np.repeat(value, n_c), chunks=n_c), coords={C: cs}, name=name)


def _get_shape_dim(image: xr.DataArray, dim: str) -> int:
    """Get the shape of a dimension in an image.

    Parameters
    ----------
    image : xr.DataArray
        The image to get the shape of.
    dim : str
        The dimension to get the shape of.

    Returns
    -------
    int
        The shape of the dimension.
    """
    (s,) = image.coords[dim].shape
    return s


def _convert_kwarg_to_xr_vec(
    bound: BoundArguments,
    *arg_names: P.args,
) -> xr.DataArray:
    """Convert a specified argument to an xr.DataArray if it is a scalar.

    Parameters
    ----------
    bound : BoundArguments
        The bound arguments of the function.
    *arg_names : str
        The name of the argument to convert.

    Returns
    -------
    xr.DataArray
        The converted argument.
    """
    cs = get_channels(bound.arguments["image"])
    n_c = _get_shape_dim(bound.arguments["image"], C)
    for arg_name in arg_names:
        if arg_name in bound.arguments and isinstance(bound.arguments[arg_name], int | float):
            bound.arguments[arg_name] = _kwarg_to_vec(value=bound.arguments[arg_name], n_c=n_c, cs=cs, name=arg_name)


def convert_kwargs_to_xr_vec(*arg_names: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Convert specified arguments to xr.DataArray if they are scalar.

    Parameters
    ----------
    *arg_names : str
        The names of the arguments to convert.

    Returns
    -------
    Callable[[Callable[..., T]], Callable[..., T]]
        A decorator that converts specified arguments to xr.DataArray if they are scalar.

    Examples
    --------
    >>> @convert_kwargs_to_xr_vec("gamma", "gain")
    ... def adjust_gamma(image: xr.DataArray, gamma: float = 1, gain: float = 1) -> xr.DataArray: ...
    """

    def kwarg_to_xr_decorator(f: Callable[P, T]) -> Callable[Concatenate[str, P], T]:
        sig = inspect.signature(f)

        @wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Callable[Concatenate[str, P], T]:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            _convert_kwarg_to_xr_vec(bound, *arg_names)
            return f(*bound.args, **bound.kwargs)

        return wrapper

    return kwarg_to_xr_decorator
