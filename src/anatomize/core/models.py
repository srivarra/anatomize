from collections.abc import Hashable, Sequence
from functools import singledispatch
from typing import TypeVar

import datatree as dt
import xarray as xr
from boost_histogram.axis import Axis
from more_itertools import first

Dimension = TypeVar("Dimension", bound=Hashable)

C: Dimension = "c"
"""The channel dimension"""
Z: Dimension = "z"
"""The z dimension"""
Y: Dimension = "y"
"""The y dimension"""
X: Dimension = "x"
"""The x dimension"""


AxisSpec = Axis | int | Sequence[int | float]
"""Accepted input types for bins specification."""


@singledispatch
def get_channels(data) -> list:
    """Get channels from data for an image element (both single and multiscale).

    Parameters
    ----------
    data
        data to get channels from

    Returns
    -------
    List of channels

    Notes
    -----
    For multiscale images, the channels are validated to be consistent across scales.
    """
    raise ValueError(f"Cannot get channels from {type(data)}")


@get_channels.register
def _(data: xr.DataArray) -> list[Hashable]:
    return data.coords[C].to_numpy().tolist()


@get_channels.register
def _(data: dt.DataTree) -> list[Hashable]:
    name = first({first(data[i].data_vars.keys()) for i in data})
    channels = {tuple(data[i][name].coords[C].to_numpy()) for i in data}
    if len(channels) > 1:
        raise ValueError(f"Channels are not consistent across scales: {channels}")
    return list(first(channels))
