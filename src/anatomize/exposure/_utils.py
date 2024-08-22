from collections.abc import Mapping

import numpy as np
import xarray as xr
from numba import guvectorize
from spatialdata.models import C, X, Y


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
