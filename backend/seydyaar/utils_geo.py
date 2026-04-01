from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np
from shapely.geometry import GeometryCollection, Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

try:
    from shapely import contains_xy as _contains_xy  # shapely >= 2
except Exception:  # pragma: no cover
    _contains_xy = None


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
        return (self.lon_max - self.lon_min) / max(self.width - 1, 1)

    @property
    def dy(self) -> float:
        return (self.lat_max - self.lat_min) / max(self.height - 1, 1)

    def lonlat_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        lons = np.linspace(self.lon_min, self.lon_max, self.width, dtype=np.float32)
        lats = np.linspace(self.lat_max, self.lat_min, self.height, dtype=np.float32)  # north->south for images
        lon2d, lat2d = np.meshgrid(lons, lats)
        return lon2d, lat2d


def _iter_polygonal_geoms(aoi_geojson: dict) -> Iterable[object]:
    if not isinstance(aoi_geojson, dict):
        return []
    t = aoi_geojson.get("type")
    if t == "FeatureCollection":
        feats = aoi_geojson.get("features") or []
        geoms = []
        for feat in feats:
            geom = (feat or {}).get("geometry")
            if not geom:
                continue
            shp = shape(geom)
            if shp.geom_type in {"Polygon", "MultiPolygon"}:
                geoms.append(shp)
        return geoms
    if t == "Feature":
        geom = aoi_geojson.get("geometry")
        if not geom:
            return []
        shp = shape(geom)
        return [shp] if shp.geom_type in {"Polygon", "MultiPolygon"} else []
    shp = shape(aoi_geojson)
    return [shp] if shp.geom_type in {"Polygon", "MultiPolygon"} else []


def _aoi_geometry(aoi_geojson: dict):
    geoms = list(_iter_polygonal_geoms(aoi_geojson))
    if not geoms:
        raise ValueError("AOI must contain at least one Polygon or MultiPolygon geometry.")
    geom = unary_union(geoms)
    if geom.is_empty:
        raise ValueError("AOI polygon union is empty.")
    minx, miny, maxx, maxy = geom.bounds
    if not (maxx > minx and maxy > miny):
        raise ValueError(f"AOI bounds collapsed to line/point: {(minx, miny, maxx, maxy)}")
    return geom


def bbox_from_geojson(aoi_geojson: dict) -> Tuple[float, float, float, float]:
    geom = _aoi_geometry(aoi_geojson)
    minx, miny, maxx, maxy = geom.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def mask_from_geojson(aoi_geojson: dict, grid: GridSpec) -> np.ndarray:
    """Returns uint8 mask (1 inside AOI, 0 outside) aligned with grid lon/lat mesh."""
    geom = _aoi_geometry(aoi_geojson)
    lon2d, lat2d = grid.lonlat_mesh()
    if _contains_xy is not None:
        try:
            mask = _contains_xy(geom, lon2d, lat2d) | _contains_xy(geom.boundary, lon2d, lat2d)
            return np.asarray(mask, dtype=np.uint8)
        except Exception:
            pass
    pg = prep(geom)
    H, W = lon2d.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            pt = Point(float(lon2d[i, j]), float(lat2d[i, j]))
            if pg.contains(pt) or geom.touches(pt):
                mask[i, j] = 1
    if not np.any(mask):
        raise ValueError("AOI mask is empty on the target grid.")
    return mask
