import pytest
from spatialdata import SpatialData
from spatialdata.datasets import blobs
from upath import UPath


@pytest.fixture(scope="session", autouse=True, name="blobs_sdata_store")
def blobs_sdata_store(tmp_path_factory: pytest.TempdirFactory) -> UPath:
    """A spatialdata blobs dataset.

    Parameters
    ----------
    tmp_path_factory
        The temporary directory factory.

    Returns
    -------
    The path to the blobs dataset.
    """
    blobs_sdata: SpatialData = blobs(length=512, n_channels=20)
    blobs_sdata_path = UPath(tmp_path_factory.mktemp("blobs_sdata") / "blobs_sdata.zarr")
    blobs_sdata.write(file_path=blobs_sdata_path)
    return blobs_sdata_path
