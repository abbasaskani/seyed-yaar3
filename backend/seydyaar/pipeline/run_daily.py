from __future__ import annotations

"""Lean scheduled pipeline for GitHub Pages hosting.

Optimization goals:
- keep the scientifically useful near-surface variables/features,
- avoid extra network fetches and repeated dataset-describe calls,
- avoid heavy/diagnostic outputs by default,
- preserve the original UI-facing core outputs and paths.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import hashlib
import json
import math
import os
import shutil
import subprocess

import numpy as np
from dateutil import parser as dtparser
from dateutil import tz
try:
    from netCDF4 import Dataset  # type: ignore
except Exception:  # pragma: no cover - optional runtime fallback
    Dataset = None  # type: ignore

from ..models.ensemble import ensemble_stats
from ..models.ocean_features import (
    anomaly,
    boa_front,
    compute_eddy_edge_distance,
    compute_eke,
    compute_okubo_weiss,
    compute_strain,
    compute_vorticity,
    detect_eddy_mask,
    front_persistence,
    fuse_fronts,
    rolling_mean,
    thermocline_proxy,
    vertical_access,
    wind_speed_dir,
)
from ..models.ops import ops_feasibility
from ..models.scoring import HabitatInputs, habitat_scoring
from ..utils_geo import GridSpec, bbox_from_geojson, mask_from_geojson
from ..utils_time import time_id_from_iso, timestamps_for_range, trusted_utc_now
from .io import minify_json_for_web, write_bin_f32, write_bin_u8, write_json


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _walk_find_key(obj: Any, key: str) -> List[Any]:
    found: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(_walk_find_key(v, key))
    elif isinstance(obj, list):
        for it in obj:
            found.extend(_walk_find_key(it, key))
    return found


class _DepthResolver:
    """Cache closest-available depth per dataset so `describe` happens only once per run."""

    def __init__(self) -> None:
        self._cache: Dict[str, Optional[float]] = {}

    def closest_depth(self, dataset_id: str, target_m: float = 0.0) -> Optional[float]:
        if dataset_id in self._cache:
            return self._cache[dataset_id]
        cmd = ["copernicusmarine", "describe", "--dataset-id", dataset_id, "-c", "depth", "-r", "coordinates"]
        try:
            cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
            meta = json.loads(cp.stdout)
        except Exception:
            self._cache[dataset_id] = None
            return None
        mins = _walk_find_key(meta, "minimum_value")
        maxs = _walk_find_key(meta, "maximum_value")
        vals: List[float] = []
        for v in mins + maxs:
            try:
                vals.append(float(v))
            except Exception:
                pass
        if not vals:
            self._cache[dataset_id] = None
            return None
        best = min(vals, key=lambda d: abs(d - target_m))
        self._cache[dataset_id] = best
        return best


_GLOBAL_DEPTH_RESOLVER = _DepthResolver()


@dataclass(frozen=True)
class RuntimeFlags:
    front_persist_3d: bool = True
    front_persist_7d: bool = False
    enable_eddy: bool = True
    enable_vertical: bool = True
    enable_chl_7d: bool = True
    enable_chl_anom: bool = False
    enable_npp_anom: bool = False
    write_extended_layers: bool = False
    write_diagnostics: bool = False
    copy_verify_nc: bool = False
    prefer_local_wind: bool = True


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _runtime_flags() -> RuntimeFlags:
    return RuntimeFlags(
        front_persist_3d=_env_flag("SEYDYAAR_ENABLE_FRONT_PERSIST_3D", True),
        front_persist_7d=_env_flag("SEYDYAAR_ENABLE_FRONT_PERSIST_7D", False),
        enable_eddy=_env_flag("SEYDYAAR_ENABLE_EDDY", True),
        enable_vertical=_env_flag("SEYDYAAR_ENABLE_VERTICAL", True),
        enable_chl_7d=_env_flag("SEYDYAAR_ENABLE_CHL_7D", True),
        enable_chl_anom=_env_flag("SEYDYAAR_ENABLE_CHL_ANOM", False),
        enable_npp_anom=_env_flag("SEYDYAAR_ENABLE_NPP_ANOM", False),
        write_extended_layers=_env_flag("SEYDYAAR_WRITE_EXTENDED_LAYERS", False),
        write_diagnostics=_env_flag("SEYDYAAR_WRITE_DIAGNOSTICS", False),
        copy_verify_nc=_env_flag("SEYDYAAR_COPY_VERIFY_NC", False),
        prefer_local_wind=_env_flag("SEYDYAAR_PREFER_LOCAL_WIND", True),
    )


def _seed_from_ts(ts_iso: str) -> int:
    h = 2166136261
    for ch in ts_iso.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _get_copernicus_creds() -> Tuple[str, str]:
    user = os.getenv("COPERNICUS_MARINE_USERNAME", "").strip()
    pwd = os.getenv("COPERNICUS_MARINE_PASSWORD", "").strip()
    if not user:
        user = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME", "").strip()
    if not pwd:
        pwd = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD", "").strip()
    return user, pwd


def _dataset_offsets_hours(key: str) -> List[int]:
    # Narrow the search range to the cadence that makes sense for the dataset family.
    if key in {"chl", "mld", "o2", "npp"}:
        return [0, -24, 24, -48, 48]
    if key == "sss":
        return [0, -6, 6, -12, 12, -24, 24]
    return [0, -6, 6, -12, 12, -24, 24]


def _synthetic_env_layers(grid: GridSpec, ts_iso: str) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(_seed_from_ts(ts_iso))
    lon2d, lat2d = grid.lonlat_mesh()
    sst = 26.0 + 2.0 * np.sin((lat2d - lat2d.mean()) * math.pi / 15.0) + 0.7 * np.cos((lon2d - lon2d.mean()) * math.pi / 20.0)
    sst += rng.normal(0, 0.25, size=sst.shape)
    chl = 0.2 + 0.08 * np.cos((lat2d - lat2d.mean()) * math.pi / 10.0) + 0.05 * np.sin((lon2d - lon2d.mean()) * math.pi / 12.0)
    chl = np.clip(chl + rng.normal(0, 0.01, size=chl.shape), 0.02, 2.0)
    ssh = 0.2 * np.sin((lon2d - lon2d.mean()) * math.pi / 8.0) * np.cos((lat2d - lat2d.mean()) * math.pi / 8.0)
    ssh += rng.normal(0, 0.01, size=ssh.shape)
    u = 0.25 + 0.12 * np.cos((lat2d - lat2d.mean()) * math.pi / 13.0) + rng.normal(0, 0.02, size=sst.shape)
    v = 0.18 + 0.10 * np.sin((lon2d - lon2d.mean()) * math.pi / 11.0) + rng.normal(0, 0.02, size=sst.shape)
    cur = np.sqrt(u * u + v * v)
    waves = 1.1 + 0.4 * np.cos((lat2d - lat2d.mean()) * math.pi / 14.0)
    waves = np.clip(waves + rng.normal(0, 0.05, size=waves.shape), 0.0, 4.0)
    mld = np.clip(20.0 + 15.0 * np.sin((lat2d - lat2d.mean()) * math.pi / 12.0) + rng.normal(0, 2.0, size=sst.shape), 5.0, 120.0)
    o2 = np.clip(180.0 + 20.0 * np.cos((lon2d - lon2d.mean()) * math.pi / 16.0) + rng.normal(0, 4.0, size=sst.shape), 80.0, 260.0)
    sss = np.clip(35.2 + 0.5 * np.sin((lon2d - lon2d.mean()) * math.pi / 18.0) + rng.normal(0, 0.05, size=sst.shape), 32.0, 37.5)
    npp = np.clip(0.8 + 0.4 * np.cos((lat2d - lat2d.mean()) * math.pi / 11.0) + rng.normal(0, 0.05, size=sst.shape), 0.05, 3.0)
    wind_u10 = 4.5 + 2.0 * np.cos((lat2d - lat2d.mean()) * math.pi / 14.0) + rng.normal(0, 0.3, size=sst.shape)
    wind_v10 = 1.0 + 1.5 * np.sin((lon2d - lon2d.mean()) * math.pi / 14.0) + rng.normal(0, 0.3, size=sst.shape)
    qc_chl = (rng.random(size=chl.shape) > 0.07).astype(np.uint8)
    conf = qc_chl.astype(np.float32)
    return {
        "sst_c": sst.astype(np.float32),
        "chl_mg_m3": chl.astype(np.float32),
        "ssh_m": ssh.astype(np.float32),
        "u_current_m_s": u.astype(np.float32),
        "v_current_m_s": v.astype(np.float32),
        "current_m_s": cur.astype(np.float32),
        "waves_hs_m": waves.astype(np.float32),
        "mld_m": mld.astype(np.float32),
        "o2_mmol_m3": o2.astype(np.float32),
        "sss_psu": sss.astype(np.float32),
        "npp_mgC_m3_d": npp.astype(np.float32),
        "wind_u10_m_s": wind_u10.astype(np.float32),
        "wind_v10_m_s": wind_v10.astype(np.float32),
        "qc_chl": qc_chl,
        "conf": conf,
        "wind_source": np.array([0], dtype=np.uint8),
    }


def _squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 2:
        out = out[0]
    if out.ndim != 2:
        raise RuntimeError(f"Expected 2D array after slicing, got shape {out.shape}")
    return np.asarray(out, dtype=np.float32)


def _read_nc_vars(path: Path, variables: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if Dataset is not None:
        with Dataset(path.as_posix(), mode="r") as ds:  # type: ignore[misc]
            for name in variables:
                if name not in ds.variables:
                    raise KeyError(f"Variable '{name}' not found in {path.name}")
                v = ds.variables[name]
                arr = _squeeze_to_2d(v[...])
                fill = getattr(v, "_FillValue", None)
                if fill is None:
                    fill = getattr(v, "missing_value", None)
                if fill is not None:
                    try:
                        arr[arr == np.float32(fill)] = np.nan
                    except Exception:
                        pass
                arr[~np.isfinite(arr)] = np.nan
                arr[np.abs(arr) > np.float32(1e6)] = np.nan
                out[name] = arr.astype(np.float32)
        return out

    import rasterio
    for name in variables:
        with rasterio.open(f'NETCDF:"{path}":{name}') as ds:
            arr = ds.read(1).astype(np.float32)
            nodata = ds.nodata
        if nodata is not None:
            arr[arr == np.float32(nodata)] = np.nan
        arr[~np.isfinite(arr)] = np.nan
        arr[np.abs(arr) > np.float32(1e6)] = np.nan
        out[name] = arr.astype(np.float32)
    return out


def _resize_bilinear_nan(a: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = a.shape
    arr = np.asarray(a, dtype=np.float32)
    if src_h == target_h and src_w == target_w:
        return arr.astype(np.float32, copy=False)

    y = np.linspace(0.0, src_h - 1, target_h, dtype=np.float32)
    x = np.linspace(0.0, src_w - 1, target_w, dtype=np.float32)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)

    out = np.empty((target_h, target_w), dtype=np.float32)
    for i in range(target_h):
        top0 = arr[y0[i], x0]
        top1 = arr[y0[i], x1]
        bot0 = arr[y1[i], x0]
        bot1 = arr[y1[i], x1]

        w00 = (1.0 - wy[i]) * (1.0 - wx)
        w01 = (1.0 - wy[i]) * wx
        w10 = wy[i] * (1.0 - wx)
        w11 = wy[i] * wx

        vals = np.stack([top0, top1, bot0, bot1], axis=0)
        ws = np.stack([w00, w01, w10, w11], axis=0).astype(np.float32)
        valid = np.isfinite(vals)
        num = np.sum(np.where(valid, vals * ws, 0.0), axis=0)
        den = np.sum(np.where(valid, ws, 0.0), axis=0)
        row = num / np.maximum(den, 1e-9)
        row[den <= 1e-9] = np.nan
        out[i] = row.astype(np.float32)
    return out


def _load_local_era5_wind(grid: GridSpec, ts_iso: str, template: str, variables: Sequence[str]) -> Optional[Dict[str, np.ndarray]]:
    date_key = dtparser.isoparse(ts_iso).astimezone(tz.UTC).strftime("%Y%m%d")
    path = Path(template.format(date=date_key))
    if not path.exists():
        return None
    try:
        names = list(variables)[:2] if len(list(variables)) >= 2 else ["u10", "v10"]
        vv = _read_nc_vars(path, names)
        u = _resize_bilinear_nan(vv[names[0]], grid.height, grid.width)
        v = _resize_bilinear_nan(vv[names[1]], grid.height, grid.width)
        return {
            "wind_u10_m_s": u.astype(np.float32),
            "wind_v10_m_s": v.astype(np.float32),
            "wind_source": np.array([1], dtype=np.uint8),
        }
    except Exception:
        return None


def _wind_proxy_from_surface(layers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # Cheap but stable proxy: no extra network fetch, direction from surface current, speed from waves+current.
    u = np.asarray(layers.get("u_current_m_s"), np.float32)
    v = np.asarray(layers.get("v_current_m_s"), np.float32)
    cur = np.asarray(layers.get("current_m_s"), np.float32)
    waves = np.asarray(layers.get("waves_hs_m"), np.float32)
    direction = np.arctan2(v, u)
    speed = np.clip(2.0 * waves + 1.5 * cur, 0.0, 18.0).astype(np.float32)
    return {
        "wind_u10_m_s": (speed * np.cos(direction)).astype(np.float32),
        "wind_v10_m_s": (speed * np.sin(direction)).astype(np.float32),
        "wind_source": np.array([2], dtype=np.uint8),
    }


def _try_copernicus_layers(
    grid: GridSpec,
    bbox: Tuple[float, float, float, float],
    ts_iso: str,
    datasets_cfg: Dict[str, Any],
    flags: RuntimeFlags,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    if isinstance(datasets_cfg, dict) and "cmems" in datasets_cfg and isinstance(datasets_cfg["cmems"], dict):
        datasets_cfg = datasets_cfg["cmems"]

    user, pwd = _get_copernicus_creds()
    status: Dict[str, Any] = {"provider": "copernicusmarine", "ok": False, "errors": [], "warnings": []}
    if not (user and pwd):
        status["errors"].append("missing Copernicus credentials")
        return None, status

    try:
        import copernicusmarine  # type: ignore
    except Exception as e:
        status["errors"].append(f"copernicusmarine import failed: {e}")
        return None, status

    for k in ["sst", "chl", "ssh", "currents", "waves"]:
        if not str(datasets_cfg.get(k, {}).get("dataset_id", "")).strip():
            status["errors"].append(f"datasets.json missing dataset_id for '{k}'")
            return None, status

    tmpdir = Path(os.getenv("SEYDYAAR_TMPDIR", ".seydyaar_tmp"))
    tmpdir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(os.getenv("SEYDYAAR_LOG_DIR", "docs/latest/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = log_dir / "download_manifest.jsonl"

    lon_min, lat_min, lon_max, lat_max = bbox
    t0 = dtparser.isoparse(ts_iso).astimezone(tz.UTC)

    def _vnames(key: str) -> List[str]:
        cfg = datasets_cfg[key]
        vs = cfg.get("variables") or []
        if vs:
            return list(vs)
        v = cfg.get("variable")
        return [v] if v else []

    def _subset_one(key: str) -> Path:
        cfg = datasets_cfg[key]
        dsid = cfg["dataset_id"]
        vars_ = _vnames(key)
        if not vars_:
            raise RuntimeError(f"{key}: variables list is empty in datasets.json")
        last_err: Optional[Exception] = None
        for off in _dataset_offsets_hours(key):
            tt0 = t0 + timedelta(hours=off)
            tt1 = tt0
            p = tmpdir / f"{key}_{tt0.strftime('%Y%m%dT%H%M%S')}.nc"
            rec: Dict[str, Any] = {
                "layer": key,
                "dataset_id": dsid,
                "variables": vars_,
                "requested_time_utc": t0.isoformat(),
                "resolved_time_utc": tt0.isoformat(),
                "bbox": [float(lon_min), float(lat_min), float(lon_max), float(lat_max)],
                "coordinates_selection_method": "nearest",
                "depth_target_m": cfg.get("depth_target_m", cfg.get("depth_m", None)),
                "depth_selected_m": None,
                "output_nc": str(p),
                "ok": False,
                "bytes": 0,
                "sha256": None,
                "error": None,
                "reused": False,
            }
            try:
                depth_target = cfg.get("depth_target_m", cfg.get("depth_m", None))
                min_depth = max_depth = None
                if depth_target is not None:
                    try:
                        best = _GLOBAL_DEPTH_RESOLVER.closest_depth(dsid, target_m=float(depth_target))
                        if best is not None:
                            rec["depth_selected_m"] = float(best)
                            min_depth = max_depth = float(best)
                    except Exception:
                        pass
                if p.exists() and p.stat().st_size > 0:
                    rec["ok"] = True
                    rec["reused"] = True
                    rec["bytes"] = p.stat().st_size
                    rec["sha256"] = _sha256_file(p)
                    _append_jsonl(manifest_path, rec)
                    status.setdefault("resolved_times", {})[key] = tt0.isoformat()
                    status.setdefault("nc_paths", {})[key] = str(p)
                    return p
                copernicusmarine.subset(
                    dataset_id=dsid,
                    variables=vars_,
                    minimum_longitude=lon_min,
                    maximum_longitude=lon_max,
                    minimum_latitude=lat_min,
                    maximum_latitude=lat_max,
                    minimum_depth=min_depth,
                    maximum_depth=max_depth,
                    start_datetime=tt0.isoformat(),
                    end_datetime=tt1.isoformat(),
                    username=user,
                    password=pwd,
                    output_filename=str(p),
                    overwrite=True,
                    skip_existing=False,
                    coordinates_selection_method="nearest",
                )
                status.setdefault("resolved_times", {})[key] = tt0.isoformat()
                status.setdefault("nc_paths", {})[key] = str(p)
                if p.exists():
                    rec["ok"] = True
                    rec["bytes"] = p.stat().st_size
                    rec["sha256"] = _sha256_file(p)
                _append_jsonl(manifest_path, rec)
                return p
            except Exception as e:
                rec["error"] = str(e)
                _append_jsonl(manifest_path, rec)
                last_err = e
                continue
        raise RuntimeError(f"{key}: subset failed for {t0.isoformat()} (tried dataset-cadence offsets). Last error: {last_err}")

    def _to_grid(a: np.ndarray) -> np.ndarray:
        return _resize_bilinear_nan(a, grid.height, grid.width)

    out: Dict[str, np.ndarray] = {}
    try:
        p_sst = _subset_one("sst")
        out["sst_c"] = _to_grid(_read_nc_vars(p_sst, _vnames("sst"))[_vnames("sst")[0]])

        p_chl = _subset_one("chl")
        out["chl_mg_m3"] = _to_grid(_read_nc_vars(p_chl, _vnames("chl"))[_vnames("chl")[0]])

        p_ssh = _subset_one("ssh")
        out["ssh_m"] = _to_grid(_read_nc_vars(p_ssh, _vnames("ssh"))[_vnames("ssh")[0]])

        p_cur = _subset_one("currents")
        cur_names = _vnames("currents")
        vv = _read_nc_vars(p_cur, cur_names[:2])
        u = _to_grid(vv[cur_names[0]])
        v = _to_grid(vv[cur_names[1]])
        out["u_current_m_s"] = u.astype(np.float32)
        out["v_current_m_s"] = v.astype(np.float32)
        out["current_m_s"] = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2).astype(np.float32)

        p_waves = _subset_one("waves")
        out["waves_hs_m"] = _to_grid(_read_nc_vars(p_waves, _vnames("waves"))[_vnames("waves")[0]])

        if flags.enable_vertical:
            for key, out_key in (("sss", "sss_psu"), ("mld", "mld_m"), ("o2", "o2_mmol_m3")):
                cfg = datasets_cfg.get(key, {})
                if not str(cfg.get("dataset_id", "")).strip():
                    continue
                try:
                    p_opt = _subset_one(key)
                    var_name = _vnames(key)[0]
                    out[out_key] = _to_grid(_read_nc_vars(p_opt, [var_name])[var_name])
                except Exception as ee:
                    status["warnings"].append(f"{key} optional layer skipped: {ee}")

        if flags.enable_npp_anom:
            cfg = datasets_cfg.get("npp", {})
            if str(cfg.get("dataset_id", "")).strip():
                try:
                    p_npp = _subset_one("npp")
                    var_name = _vnames("npp")[0]
                    out["npp_mgC_m3_d"] = _to_grid(_read_nc_vars(p_npp, [var_name])[var_name])
                except Exception as ee:
                    status["warnings"].append(f"npp optional layer skipped: {ee}")

        # Cheap wind path: local ERA5 if available, otherwise proxy. No extra online fetch by default.
        wind_cfg = datasets_cfg.get("wind", {}) if isinstance(datasets_cfg, dict) else {}
        wind_loaded: Optional[Dict[str, np.ndarray]] = None
        if flags.prefer_local_wind:
            template = str(wind_cfg.get("local_template", "backend/data/wind/era5_{date}.nc"))
            variables = wind_cfg.get("variables") or ["u10", "v10"]
            wind_loaded = _load_local_era5_wind(grid, ts_iso, template, variables)
            if wind_loaded is None:
                status["warnings"].append("local ERA5 wind not found; using wave/current proxy")
        if wind_loaded is None:
            wind_loaded = _wind_proxy_from_surface(out)
        out.update(wind_loaded)

        qc = np.ones((grid.height, grid.width), dtype=np.uint8)
        conf = qc.astype(np.float32)
        out["qc_chl"] = qc
        out["conf"] = conf
        status["ok"] = True
        return out, status
    except Exception as e:
        status["errors"].append(str(e))
        return None, status


def _write_meta_index(out_root: Path, run_entry: Dict[str, Any]) -> None:
    idx_path = out_root / "meta_index.json"
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            idx = {"version": 1, "runs": []}
    else:
        idx = {"version": 1, "runs": []}
    idx["runs"] = [r for r in idx.get("runs", []) if r.get("run_id") != run_entry["run_id"]] + [run_entry]
    idx["runs"] = sorted(idx["runs"], key=lambda r: r.get("generated_at_utc", ""))
    idx["latest_run_id"] = run_entry["run_id"]
    now_utc, _ = trusted_utc_now()
    idx["generated_at_utc"] = now_utc.isoformat().replace("+00:00", "Z")
    write_json(idx_path, idx)
    minify_json_for_web(idx_path)


def _write_latest_index_and_meta(out_root: Path, run_entry: Dict[str, Any], variant: str) -> None:
    run_root = out_root / run_entry.get("path", "")
    run_meta_path = run_root / "meta.json"
    run_meta = None
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            run_meta = None
    time_ids = (run_meta or {}).get("available_time_ids") or []
    latest_tid = (run_meta or {}).get("latest_available_time_id") or (time_ids[-1] if time_ids else None)
    now_utc, _ = trusted_utc_now()
    gen = now_utc.isoformat().replace("+00:00", "Z")
    index = {
        "version": 1,
        "schema": "seydyaar-latest-index-v1",
        "generated_at_utc": gen,
        "latest_run_id": run_entry.get("run_id"),
        "run_path": run_entry.get("path"),
        "variant_default": variant,
        "species": run_entry.get("species", []),
        "models": run_entry.get("models", []),
        "time_count": len(time_ids),
        "available_time_ids": time_ids,
        "latest_available_time_id": latest_tid,
        "notes": "Compatibility endpoint. Raw outputs live under runs/<run_id>/variants/...",
    }
    idx_out = out_root / "index.json"
    write_json(idx_out, index)
    minify_json_for_web(idx_out)
    meta = {
        "version": 1,
        "generated_at_utc": gen,
        "run_id": run_entry.get("run_id"),
        "variant": variant,
        "time_source": (run_meta or {}).get("time_source"),
        "latest_available_time_id": latest_tid,
        "grid": (run_meta or {}).get("grid"),
        "bbox": (run_meta or {}).get("bbox"),
        "aoi": (run_meta or {}).get("aoi"),
        "species": run_entry.get("species", []),
        "models": run_entry.get("models", []),
        "available_time_ids": time_ids,
    }
    meta_out = out_root / "meta.json"
    write_json(meta_out, meta)
    minify_json_for_web(meta_out)


def run_daily(
    out_root: Path,
    aoi_geojson: dict,
    species_profiles: dict,
    date: str = "today",
    past_days: int = 1,
    future_days: int = 5,
    step_hours: int = 12,
    grid_wh: str = "160x160",
    variant: str = "auto",
    gear_depths_m: List[int] = [5, 10, 15, 20],
    species_filter: Optional[List[str]] = None,
) -> str:
    flags = _runtime_flags()
    now_utc, time_source = trusted_utc_now()
    anchor = now_utc.date() if date.lower() == "today" else datetime.fromisoformat(date).date()
    step_hours = max(int(step_hours), 6)
    run_id = "main"
    W, H = [int(x) for x in grid_wh.lower().split("x")]
    bbox = bbox_from_geojson(aoi_geojson)
    grid = GridSpec(lon_min=bbox[0], lat_min=bbox[1], lon_max=bbox[2], lat_max=bbox[3], width=W, height=H)
    mask = mask_from_geojson(aoi_geojson, grid)
    ts_list = timestamps_for_range(anchor_date=date, past_days=past_days, future_days=future_days, step_hours=step_hours)
    time_ids = [time_id_from_iso(iso) for iso in ts_list]
    id_by_iso = {iso: tid for iso, tid in zip(ts_list, time_ids)}
    run_root = out_root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    datasets_cfg_path = Path("backend/config/datasets.json")
    datasets_cfg = json.loads(datasets_cfg_path.read_text(encoding="utf-8")) if datasets_cfg_path.exists() else {}
    if isinstance(datasets_cfg, dict) and "cmems" in datasets_cfg and isinstance(datasets_cfg["cmems"], dict):
        datasets_cfg = datasets_cfg["cmems"]

    selected_profiles = dict(species_profiles)
    if species_filter:
        wanted = {s.strip() for s in species_filter if s.strip()}
        selected_profiles = {k: v for k, v in species_profiles.items() if k in wanted}
    if not selected_profiles:
        raise RuntimeError("No species selected after applying species_filter")

    run_meta = {
        "run_id": run_id,
        "date": anchor.isoformat(),
        "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
        "time_source": time_source,
        "times": ts_list,
        "time_ids": time_ids,
        "variants": [variant],
        "species": list(selected_profiles.keys()),
        "bbox": list(bbox),
        "step_hours": step_hours,
        "grid": {"width": W, "height": H, "lon_min": grid.lon_min, "lon_max": grid.lon_max, "lat_min": grid.lat_min, "lat_max": grid.lat_max},
        "runtime_flags": flags.__dict__,
    }
    write_json(run_root / "meta.json", run_meta)
    minify_json_for_web(run_root / "meta.json")

    strict_cmems = os.getenv("SEYDYAAR_STRICT_COPERNICUS", "0") == "1"
    force = os.getenv("SEYDYAAR_FORCE_REGEN", "0") == "1"
    verify_dir = Path(os.getenv("SEYDYAAR_VERIFY_DIR", out_root / "verify"))
    if flags.copy_verify_nc:
        verify_dir.mkdir(parents=True, exist_ok=True)
        verify_time_id = now_utc.replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y%m%d_0000Z")
    else:
        verify_time_id = None

    # Fetch each timestamp exactly once, then reuse for all species and for lag/persistence features.
    layers_by_tid: Dict[str, Dict[str, np.ndarray]] = {}
    provider_status_by_tid: Dict[str, Dict[str, Any]] = {}
    for ts_iso in ts_list:
        tid = id_by_iso[ts_iso]
        layers, status = _try_copernicus_layers(grid, bbox, ts_iso, datasets_cfg, flags) if datasets_cfg else (None, {"provider": "none", "ok": False, "errors": ["no datasets.json"]})
        if layers is None:
            if strict_cmems:
                raise RuntimeError("Copernicus download failed (strict mode): " + "; ".join(status.get("errors", [])))
            layers = _synthetic_env_layers(grid, ts_iso)
            status = {**status, "fallback": "synthetic"}
        layers_by_tid[tid] = layers
        provider_status_by_tid[tid] = status
        if flags.copy_verify_nc and verify_time_id and tid == verify_time_id:
            nc_paths = status.get("nc_paths") or {}
            dest = verify_dir / verify_time_id
            dest.mkdir(parents=True, exist_ok=True)
            for k, src in nc_paths.items():
                try:
                    sp = Path(src)
                    if sp.exists():
                        shutil.copy2(sp, dest / f"{k}.nc")
                except Exception:
                    pass

    front3_steps = max(1, round(72 / step_hours))
    front7_steps = max(1, round(168 / step_hours))
    chl3_steps = front3_steps
    chl7_steps = front7_steps

    for sp, prof in selected_profiles.items():
        priors = prof.get("priors", {})
        weights = prof.get("layer_weights", {})
        ops_priors = {**priors, **prof.get("ops_constraints", {})}
        front_fusion_weights = priors.get("front_fusion_weights", {})

        sp_root = run_root / "variants" / variant / "species" / sp
        times_root = sp_root / "times"
        times_root.mkdir(parents=True, exist_ok=True)
        write_bin_u8(sp_root / "mask_u8.bin", mask)

        sp_meta = {
            "species": sp,
            "label": prof.get("label", {}),
            "grid": run_meta["grid"],
            "times": ts_list,
            "time_ids": time_ids,
            "paths": {
                "mask": f"variants/{variant}/species/{sp}/mask_u8.bin",
                "per_time": {
                    "pcatch_scoring": f"variants/{variant}/species/{sp}/times/{{time}}/pcatch_scoring_f32.bin",
                    "pcatch_frontplus": f"variants/{variant}/species/{sp}/times/{{time}}/pcatch_frontplus_f32.bin",
                    "pcatch_ensemble": f"variants/{variant}/species/{sp}/times/{{time}}/pcatch_ensemble_f32.bin",
                    "phab_scoring": f"variants/{variant}/species/{sp}/times/{{time}}/phab_f32.bin",
                    "phab_frontplus": f"variants/{variant}/species/{sp}/times/{{time}}/phab_frontplus_f32.bin",
                    "pops": f"variants/{variant}/species/{sp}/times/{{time}}/pops_f32.bin",
                    "agree": f"variants/{variant}/species/{sp}/times/{{time}}/agree_f32.bin",
                    "spread": f"variants/{variant}/species/{sp}/times/{{time}}/spread_f32.bin",
                    "front": f"variants/{variant}/species/{sp}/times/{{time}}/front_f32.bin",
                    "sst": f"variants/{variant}/species/{sp}/times/{{time}}/sst_f32.bin",
                    "chl": f"variants/{variant}/species/{sp}/times/{{time}}/chl_f32.bin",
                    "current": f"variants/{variant}/species/{sp}/times/{{time}}/current_f32.bin",
                    "waves": f"variants/{variant}/species/{sp}/times/{{time}}/waves_f32.bin",
                    "conf": f"variants/{variant}/species/{sp}/times/{{time}}/conf_f32.bin",
                    "qc_chl": f"variants/{variant}/species/{sp}/times/{{time}}/qc_chl_u8.bin"
                }
            },
            "model_info": {
                "habitat": {"priors": priors, "weights": weights},
                "ops": {"priors": ops_priors, "gear_depths_m": gear_depths_m},
                "runtime_flags": flags.__dict__,
            },
        }
        write_json(sp_root / "meta.json", sp_meta)
        minify_json_for_web(sp_root / "meta.json")

        provider_status: List[Dict[str, Any]] = []
        front_base_q: Deque[np.ndarray] = deque(maxlen=max(front7_steps, front3_steps, 1))

        for ts_iso in ts_list:
            tid = id_by_iso[ts_iso]
            tdir = times_root / tid
            if (not force) and (tdir / "pcatch_scoring_f32.bin").exists():
                provider_status.append({"timestamp": ts_iso, "skipped": True, "reason": "already_exists"})
                continue
            layers = layers_by_tid[tid]
            status = provider_status_by_tid[tid]
            provider_status.append({"timestamp": ts_iso, **status})

            sst = layers["sst_c"]
            chl = layers["chl_mg_m3"]
            ssh = layers["ssh_m"]
            cur = layers["current_m_s"]
            waves = layers["waves_hs_m"]
            ucur = layers["u_current_m_s"]
            vcur = layers["v_current_m_s"]

            front_sst = boa_front(sst, denoise_radius=1, background_radius=3)
            front_logchl = boa_front(np.log10(np.clip(chl, 1e-6, None)), denoise_radius=1, background_radius=3)
            front_ssh = boa_front(ssh, denoise_radius=1, background_radius=2)
            base_front = fuse_fronts(front_sst, front_logchl, front_ssh, None, None, {
                "sst": front_fusion_weights.get("sst", 0.34),
                "chl": front_fusion_weights.get("chl", 0.28),
                "ssh": front_fusion_weights.get("ssh", 0.18),
            })
            front_base_q.append(base_front)
            persist3 = front_persistence(list(front_base_q)[-front3_steps:]) if flags.front_persist_3d else None
            persist7 = front_persistence(list(front_base_q)[-front7_steps:]) if flags.front_persist_7d else None
            front_fused = fuse_fronts(front_sst, front_logchl, front_ssh, persist3, persist7, front_fusion_weights)

            if flags.enable_vertical and all(k in layers for k in ("mld_m", "o2_mmol_m3", "sss_psu")):
                vert, _ = vertical_access(layers["mld_m"], layers["o2_mmol_m3"], layers["sss_psu"])
                thermo = thermocline_proxy(layers["mld_m"])
            else:
                vert = None
                thermo = None

            chl_3d = rolling_mean(layers_by_tid, time_ids, tid, "chl_mg_m3", chl3_steps)
            chl_7d = rolling_mean(layers_by_tid, time_ids, tid, "chl_mg_m3", chl7_steps) if flags.enable_chl_7d else None
            chl_anom = anomaly(chl, chl_7d) if (flags.enable_chl_anom and chl_7d is not None) else None
            npp_anom = None
            if flags.enable_npp_anom and "npp_mgC_m3_d" in layers:
                npp_base = rolling_mean(layers_by_tid, time_ids, tid, "npp_mgC_m3_d", chl7_steps)
                npp_anom = anomaly(layers["npp_mgC_m3_d"], npp_base)

            if flags.enable_eddy:
                vort = compute_vorticity(ucur, vcur)
                strain = compute_strain(ucur, vcur)
                ow = compute_okubo_weiss(vort, strain)
                eke = compute_eke(ucur, vcur)
                eddy_mask = detect_eddy_mask(ow, ssh)
                eddy_edge = compute_eddy_edge_distance(eddy_mask)
            else:
                vort = strain = ow = eke = eddy_edge = None

            ws, wd = wind_speed_dir(layers["wind_u10_m_s"], layers["wind_v10_m_s"])
            inputs = HabitatInputs(
                sst_c=sst,
                chl_mg_m3=chl,
                current_m_s=cur,
                waves_hs_m=waves,
                ssh_m=ssh,
                front_fused=front_fused,
                eke=eke,
                vorticity=vort if flags.write_extended_layers else None,
                strain=strain if flags.write_extended_layers else None,
                okubo_weiss=ow,
                eddy_edge_distance=eddy_edge,
                vertical_access=vert,
                chl_3d_mean=chl_3d,
                chl_7d_mean=chl_7d,
                chl_anom=chl_anom,
                npp_anom=npp_anom,
                thermocline_proxy=thermo,
            )
            phab, _ = habitat_scoring(inputs, priors=priors, weights=weights)
            pops = ops_feasibility(cur, waves, ops_priors, gear_depth_m=10.0, wind_speed_m_s=ws)
            pcatch = np.clip(phab * pops, 0.0, 1.0).astype(np.float32)
            front_mult = np.clip(0.96 + 0.10 * front_fused, 0.95, 1.08).astype(np.float32)
            frontplus = np.clip(pcatch * front_mult, 0.0, 1.0).astype(np.float32)
            phab_frontplus = np.clip(phab * front_mult, 0.0, 1.0).astype(np.float32)
            ens = np.nanmean(np.stack([pcatch, frontplus], axis=0), axis=0).astype(np.float32)
            agree, spread = ensemble_stats([pcatch, frontplus])

            tdir.mkdir(parents=True, exist_ok=True)
            write_bin_f32(tdir / "pcatch_scoring_f32.bin", pcatch)
            write_bin_f32(tdir / "pcatch_frontplus_f32.bin", frontplus)
            write_bin_f32(tdir / "pcatch_ensemble_f32.bin", ens)
            write_bin_f32(tdir / "phab_f32.bin", phab)
            write_bin_f32(tdir / "phab_frontplus_f32.bin", phab_frontplus)
            write_bin_f32(tdir / "pops_f32.bin", pops)
            write_bin_f32(tdir / "agree_f32.bin", agree)
            write_bin_f32(tdir / "spread_f32.bin", spread)
            write_bin_f32(tdir / "front_f32.bin", front_fused)
            write_bin_f32(tdir / "sst_f32.bin", sst.astype(np.float32))
            write_bin_f32(tdir / "chl_f32.bin", chl.astype(np.float32))
            write_bin_f32(tdir / "current_f32.bin", cur.astype(np.float32))
            write_bin_f32(tdir / "waves_f32.bin", waves.astype(np.float32))
            write_bin_u8(tdir / "qc_chl_u8.bin", layers["qc_chl"])
            write_bin_f32(tdir / "conf_f32.bin", layers["conf"])

            if flags.write_extended_layers:
                if eke is not None:
                    write_bin_f32(tdir / "eke_f32.bin", eke)
                if ow is not None:
                    write_bin_f32(tdir / "okubo_weiss_f32.bin", ow)
                if eddy_edge is not None:
                    write_bin_f32(tdir / "eddy_edge_distance_f32.bin", eddy_edge)
                if vert is not None:
                    write_bin_f32(tdir / "vertical_access_f32.bin", vert)
                write_bin_f32(tdir / "wind_speed_f32.bin", ws)
                write_bin_f32(tdir / "wind_direction_f32.bin", wd)
                if flags.write_diagnostics:
                    write_bin_f32(tdir / "front_boa_sst_f32.bin", front_sst)
                    write_bin_f32(tdir / "front_boa_logchl_f32.bin", front_logchl)
                    write_bin_f32(tdir / "front_ssh_f32.bin", front_ssh)
                    if persist3 is not None:
                        write_bin_f32(tdir / "front_persist_3d_f32.bin", persist3)
                    if persist7 is not None:
                        write_bin_f32(tdir / "front_persist_7d_f32.bin", persist7)
                    if chl_3d is not None:
                        write_bin_f32(tdir / "chl_3d_mean_f32.bin", chl_3d)
                    if chl_7d is not None:
                        write_bin_f32(tdir / "chl_7d_mean_f32.bin", chl_7d)
                    if chl_anom is not None:
                        write_bin_f32(tdir / "chl_anom_f32.bin", chl_anom)
                    if npp_anom is not None:
                        write_bin_f32(tdir / "npp_anom_f32.bin", npp_anom)
                    if vort is not None:
                        write_bin_f32(tdir / "vorticity_f32.bin", vort)
                    if strain is not None:
                        write_bin_f32(tdir / "strain_f32.bin", strain)
                    if thermo is not None:
                        write_bin_f32(tdir / "thermocline_proxy_f32.bin", thermo)

        sp_meta2 = json.loads((sp_root / "meta.json").read_text(encoding="utf-8"))
        sp_meta2["provider_status"] = provider_status
        write_json(sp_root / "meta.json", sp_meta2)
        minify_json_for_web(sp_root / "meta.json")

    run_entry = {
        "run_id": run_id,
        "path": f"runs/{run_id}",
        "fast": False,
        "date": anchor.isoformat(),
        "time_count": len(time_ids),
        "variants": [variant],
        "species": list(selected_profiles.keys()),
        "models": ["scoring", "frontplus", "ensemble"],
        "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
    }
    # Refresh top-level meta with the final species list/time ids.
    run_meta["available_time_ids"] = time_ids
    run_meta["latest_available_time_id"] = time_ids[-1] if time_ids else None
    write_json(run_root / "meta.json", run_meta)
    minify_json_for_web(run_root / "meta.json")
    _write_meta_index(out_root, run_entry)
    _write_latest_index_and_meta(out_root, run_entry, variant)
    return run_id
