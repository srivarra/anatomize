try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from collections.abc import Sequence
from dataclasses import dataclass

import boost_histogram as bh
from dask.typing import DaskCollection


@dataclass
class QuantileKwargs(TypedDict):
    keep_attrs: bool | None = None
    skipna: bool | None = None


@dataclass
class PartitionedFactoryKwargs(TypedDict):
    histref: bh.Histogram | None = None
    axes: Sequence[bh.axis.Axis] | None = None
    storage: bh.storage.Storage | None = None
    weights: DaskCollection | None = None
    sample: DaskCollection | None = None
