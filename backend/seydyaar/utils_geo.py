from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np
from shapely.geometry import shape, Point
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


def _iter_polygon_geoms(aoi_geojson: dict) -> Iterable:
    if not isinstance(aoi_geojson, dict):
        raise ValueError("AOI must be a GeoJSON object")
    gj_type = aoi_geojson.get("type")
    if gj_type == "FeatureCollection":
        feats = aoi_geojson.get("features") or []
        for feat in feats:
            geom = shape((feat or {}).get("geometry") or {})
            if geom.geom_type in {"Polygon", "MultiPolygon"}:
                yield geom
    elif gj_type == "Feature":
        geom = shape(aoi_geojson.get("geometry") or {})
        if geom.geom_type in {"Polygon", "MultiPolygon"}:
            yield geom
    else:
        geom = shape(aoi_geojson)
        if geom.geom_type in {"Polygon", "MultiPolygon"}:
            yield geom


def _polygonal_union(aoi_geojson: dict):
    geoms = list(_iter_polygon_geoms(aoi_geojson))
    if not geoms:
        raise ValueError("AOI must contain at least one Polygon or MultiPolygon geometry")
    geom = unary_union(geoms)
    minx, miny, maxx, maxy = geom.bounds
    if not (np.isfinite(minx) and np.isfinite(miny) and np.isfinite(maxx) and np.isfinite(maxy)):
        raise ValueError("AOI bounds are not finite")
    if abs(maxx - minx) <= 1e-9 or abs(maxy - miny) <= 1e-9:
        raise ValueError("AOI bounds collapsed to a point/line; check the supplied polygon")
    return geom


def bbox_from_geojson(aoi_geojson: dict) -> Tuple[float, float, float, float]:
    geom = _polygonal_union(aoi_geojson)
    minx, miny, maxx, maxy = geom.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def mask_from_geojson(aoi_geojson: dict, grid: GridSpec) -> np.ndarray:
    """Returns uint8 mask (1 inside AOI, 0 outside) with shape (H, W)."""
    geom = _polygonal_union(aoi_geojson)
    pg = prep(geom)

    lon2d, lat2d = grid.lonlat_mesh()
    H, W = lon2d.shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            if pg.covers(Point(float(lon2d[i, j]), float(lat2d[i, j]))):
                mask[i, j] = 1
    if int(mask.sum()) == 0:
        raise ValueError("AOI mask is empty on the chosen grid; enlarge AOI or adjust grid extent")
    return mask
