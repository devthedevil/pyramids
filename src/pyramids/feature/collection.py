"""FeatureCollection — a GeoDataFrame with pyramids-specific GIS methods.

The `feature` subpackage is organised as:

- :mod:`pyramids.feature.collection` (this module) — the
  :class:`FeatureCollection` class.
- :mod:`pyramids.feature.geometry` — shape factories and coordinate
  extractors (`create_polygon`, `create_point`, `get_coords`,
  `explode_gdf`, `multi_geom_handler`, …).
- :mod:`pyramids.feature._ogr` — private OGR bridge.

CRS / EPSG / reprojection helpers (`get_epsg_from_prj`,
`reproject_coordinates`, `create_sr_from_proj`) live in
:mod:`pyramids.base.crs`. The :class:`FeatureCollection` class
exposes the most-commonly-used ones as static-method delegates for
ergonomic continuity.

`FeatureCollection` is a direct subclass of
:class:`geopandas.GeoDataFrame`. Every GeoDataFrame method
is inherited; pyramids adds rasterization, Dataset interop, vertex
extraction, and CRS-helper delegates on top. `ogr.DataSource` is
internal only; see :mod:`pyramids.feature._ogr`.
"""

from __future__ import annotations

import functools
import math
import os
import warnings
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable
from urllib.parse import urlencode

if TYPE_CHECKING:
    from pyramids.dataset import Dataset
    from pyramids.feature._lazy_collection import LazyFeatureCollection

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from osgeo import gdal, ogr
from shapely.geometry import Point, Polygon, box

from pyramids import _io as _pyramids_io
from pyramids.base._errors import (
    CRSError,
    FeatureError,
    GeometryWarning,
    InvalidGeometryError,
)
from pyramids.base._utils import Catalog, import_pyarrow, require_cleopatra
from pyramids.base.remote import is_remote
from pyramids.basemap.basemap import add_basemap
from pyramids.feature import _h3
from pyramids.feature import geometry as _geom
from pyramids.feature import tessellation as _tess
from pyramids.feature._oapif import from_ogc_features as _from_ogc_features
from pyramids.feature._wfs import from_wfs as _from_wfs

CATALOG = Catalog(raster_driver=False)

# default per-chunk batch size for `iter_features` when the
# user does not pass `chunksize`. Matches pyogrio's own
# `read_dataframe` default for row-group streaming, so the fast
# path (GeoParquet + row_group tile strategy) does not re-chunk.
_DEFAULT_ITER_BATCH_SIZE: int = 1000


# target bytes per partition when read_file(backend="dask") is
# called with neither npartitions nor chunksize. 128 MiB is
# dask-geopandas' own read_parquet default and a reasonable size for
# shapely-heavy geometry ops on modern cores. Small enough that even a
# 1 GiB shapefile gets 8 partitions; large enough to avoid one-partition-
# per-10-rows pathologies on huge feature counts.
_LAZY_TARGET_BYTES_PER_PARTITION: int = 128 * 1024 * 1024


def _resolve_lazy_partitioning(
    path: str,
    npartitions: int | None,
    chunksize: int | None,
) -> dict[str, Any]:
    """default `npartitions` from file size when not given.

    Called by `read_file(backend="dask")`. If the caller supplies
    either `npartitions` or `chunksize` we honor it verbatim. If
    they supply neither, we stat the resolved path and pick
    `npartitions = max(1, ceil(size / 128 MiB))`.

    On cloud / virtual-FS paths (`/vsi*`, `http(s)://`, `s3://`,
    etc.) `os.stat` can't size the file cheaply — there we fall
    back to `npartitions=1` rather than emit a pre-flight HEAD
    request with ambiguous semantics. Users who want more partitions
    on cloud-hosted files should pass `npartitions=` explicitly.

    Args:
        path: The already-`_to_vsi`-resolved path string.
        npartitions: User-supplied partition count, if any.
        chunksize: User-supplied rows-per-partition, if any.

    Returns:
        dict: kwargs to forward to :func:`dask_geopandas.read_file`.
        Exactly one of `npartitions` / `chunksize` is populated.
    """
    kwargs: dict[str, Any] = {}
    if npartitions is not None:
        kwargs["npartitions"] = npartitions
    elif chunksize is not None:
        kwargs["chunksize"] = chunksize
    elif path.startswith(("/vsi", "http://", "https://", "s3://", "gs://", "az://")):
        # Remote / VFS path — no cheap size probe. Fall back to 1.
        kwargs["npartitions"] = 1
    else:
        try:
            size = os.path.getsize(path)
        except OSError:
            kwargs["npartitions"] = 1
        else:
            kwargs["npartitions"] = max(
                1,
                math.ceil(size / _LAZY_TARGET_BYTES_PER_PARTITION),
            )
    return kwargs


def _require_pyarrow() -> None:
    """Raise a pyramids-branded ImportError if pyarrow is absent.

    `geopandas.read_parquet` / `GeoDataFrame.to_parquet` raise a
    generic ImportError that mentions neither `pyramids-gis` nor
    the `[parquet]` optional. This helper surfaces the install
    instruction up front so the Raises docstring is truthful.
    """
    import_pyarrow(
        "GeoParquet support requires the optional 'pyarrow' "
        "dependency. Install with one of:\n"
        "  - PyPI:        pip install 'pyramids-gis[parquet]'\n"
        "  - conda-forge: conda install -c conda-forge pyramids-parquet"
    )


# module-level LRU cache backing `FeatureCollection.list_layers`.
# Keyed on the already-resolved `str` path (post `_parse_path`). The
# tuple return type plays nicely with `functools.lru_cache` (lists are
# unhashable and would break LRU internals if returned directly).
@functools.lru_cache(maxsize=128)
def _list_layers_cached(resolved_path: str) -> tuple[str, ...]:
    """Return a tuple of layer names for a resolved path (memoised)."""
    import pyogrio

    arr = pyogrio.list_layers(resolved_path)
    return tuple(str(row[0]) for row in arr)


class FeatureCollection(GeoDataFrame):
    """A :class:`geopandas.GeoDataFrame` with pyramids-specific GIS methods.

    `FeatureColle