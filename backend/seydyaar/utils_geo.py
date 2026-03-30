from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep


@dataclass(frozen=True)
class GridSpec:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    width: int
    height: int
    crs: str = "EPSG:4326"

    @property
    def dx(self) -> float:
        return (self.lon_max - self.lon_min) / (self.width - 1)

    @property
    def dy(self) -> float:
        return (self.lat_max - self.lat_min) / (self.height - 1)

    def lonlat_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        lons = np.linspace(self.lon_min, self.lon_max, self.width, dtype=np.float32)
        lats = np.linspace(self.lat_max, self.lat_min, self.height, dtype=np.float32)  # north->south for images
        lon2d, lat2d = np.meshgrid(lons, lats)
        return lon2d, lat2d


def _iter_area_geometries(obj: dict) -> Iterable[BaseGeometry]:
    """Yield only area-bearing AOI geometries from GeoJSON.

    This intentionally ignores Point/LineString features so a mixed FeatureCollection
    with a marker first and the real polygon later still works correctly.
    """
    if not isinstance(obj, dict):
        return
    typ = obj.get("type")
    if typ == "FeatureCollection":
        for feat in obj.get("features", []):
            if isinstance(feat, dict):
                yield from _iter_area_geometries(feat)
        return
    if typ == "Feature":
        geom = obj.get("geometry")
        if isinstance(geom, dict):
            yield from _iter_area_geometries(geom)
        return
    if typ == "GeometryCollection":
        for geom in obj.get("geometries", []):
            if isinstance(geom, dict):
                yield from _iter_area_geometries(geom)
        return
    if typ not in {"Polygon", "MultiPolygon"}:
        return
    geom = shape(obj)
    if geom.is_empty:
        return
    # buffer(0) often repairs minor ring/self-intersection issues
    try:
        geom = geom.buffer(0)
    except Exception:
        pass
    if geom.is_empty:
        return
    yield geom


def polygon_aoi_from_geojson(aoi_geojson: dict) -> BaseGeometry:
    geoms = list(_iter_area_geometries(aoi_geojson))
    if not geoms:
        raise ValueError(
            "AOI must contain at least one Polygon or MultiPolygon geometry; "
            "Point/Line features are ignored."
        )
    geom = unary_union(geoms) if len(geoms) > 1 else geoms[0]
    if geom.is_empty:
        raise ValueError("AOI polygon geometry became empty after union/repair")
    minx, miny, maxx, maxy = geom.bounds
    if not all(np.isfinite([minx, miny, maxx, maxy])):
        raise ValueError("AOI bounds are non-finite")
    if not (maxx > minx and maxy > miny):
        raise ValueError(
            "AOI bounds collapsed to a point/line. "
            "Check that your AOI contains a real polygon, not only a point marker."
        )
    return geom


def bbox_from_geojson(aoi_geojson: dict) -> Tuple[float, float, float, float]:
    geom = polygon_aoi_from_geojson(aoi_geojson)
    minx, miny, maxx, maxy = geom.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def mask_from_geojson(aoi_geojson: dict, grid: GridSpec) -> np.ndarray:
    """Returns uint8 mask (1 inside AOI, 0 outside) with shape (H, W).

    Uses polygonal AOI only. Boundary pixels are included via ``covers``.
    """
    geom = polygon_aoi_from_geojson(aoi_geojson)
    pg = prep(geom)

    lon2d, lat2d = grid.lonlat_mesh()
    H, W = lon2d.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            pt = Point(float(lon2d[i, j]), float(lat2d[i, j]))
            try:
                inside = pg.covers(pt)  # type: ignore[attr-defined]
            except Exception:
                inside = geom.covers(pt)
            if inside:
                mask[i, j] = 1
    return mask
