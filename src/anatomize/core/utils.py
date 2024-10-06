from typing import TypeGuard

from dask.typing import DaskCollection


def _is_dask_collection(x: object) -> TypeGuard[DaskCollection]:
    from dask.base import is_dask_collection

    # use is_dask_collection function instead of dask.typing.DaskCollection
    # see https://github.com/pydata/xarray/pull/8241#discussion_r1476276023
    return is_dask_collection(x)
