from collections.abc import Mapping
from typing import Unpack

import xarray as xr
from xarray.core.types import QuantileMethods

from anatomize.core._typeddict_kwargs import QuantileKwargs


@xr.register_dataarray_accessor("an")
class AnatomizeAccessor:
    """XArray DataArray Accessor for anatomize."""

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def normalize(
        self,
        q: tuple[float, float] = (0.0, 1.0),
        method: QuantileMethods = "linear",
        return_quantiles: bool = False,
        chunk: Mapping[str, int] | None = None,
        **quantile_kwargs: Unpack[QuantileKwargs],
    ) -> xr.DataArray:
        """#TODO."""
        from anatomize.exposure import normalize

        return normalize(
            image=self, q=q, method=method, return_quantiles=return_quantiles, chunk=chunk, **quantile_kwargs
        )
