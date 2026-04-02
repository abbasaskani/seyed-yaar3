from __future__ import annotations

from typing import Dict, Sequence
import math
import numpy as np


def _nan_to_num(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32).copy()
    out[~np.isfinite(out)] = np.float32(fill)
    return out


def _smooth1d_nan(x: np.ndarray, radius: int) -> np.ndarray:
    vals = np.asarray(x, dtype=np.float32)
    r = max(int(radius), 0)
    if r <= 0:
        return vals.astype(np.float32, copy=True)
    valid = np.isfinite(vals)
    if not np.any(valid):
        return np.full_like(vals, np.nan, dtype=np.float32)
    vv = np.where(valid, vals, 0.0).astype(np.float32)
    ww = valid.astype(np.float32)
    vv = np.pad(vv, (r, r), mode="edge")
    ww = np.pad(ww, (r, r), mode="edge")
    sv = np.pad(vv, (1, 0), mode="constant").cumsum(0)
    sw = np.pad(ww, (1, 0), mode="constant").cumsum(0)
    k = 2 * r + 1
    sums = sv[k:] - sv[:-k]
    cnts = sw[k:] - sw[:-k]
    out = np.divide(sums, np.maximum(cnts, 1.0), dtype=np.float32)
    out[cnts <= 0.0] = np.nan
    return out.astype(np.float32)


def _safe_nanmean_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack, dtype=np.float32)
    valid = np.isfinite(arr)
    count = valid.sum(axis=0).astype(np.float32)
    num = np.where(valid, arr, 0.0).sum(axis=0, dtype=np.float32)
    out = np.divide(num, np.maximum(count, 1.0), dtype=np.float32)
    out[count <= 0.0] = np.nan
    return out.astype(np.float32)


def robust_normalize(
    a: np.ndarray,
    lo_q: float = 5.0,
    hi_q: float = 95.0,
    min_span: float = 1e-4,
) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[valid]
    lo, hi = np.nanpercentile(vals, [lo_q, hi_q])
    if not np.isfinite(lo):
        lo = float(np.nanmin(vals))
    if not np.isfinite(hi):
        hi = float(np.nanmax(vals))
    span = float(hi - lo)
    sigma = float(np.nanstd(vals))
    floor = max(float(min_span), 0.08 * sigma)
    if (not np.isfinite(span)) or span <= floor:
        out = np.full_like(arr, np.float32(0.5), dtype=np.float32)
        out[~valid] = np.nan
        return out
    out = (arr - np.float32(lo)) / np.float32(span)
    out[~valid] = np.nan
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def box_mean(arr: np.ndarray, radius: int = 1) -> np.ndarray:
    """NaN-aware 2D box mean using integral images."""
    a = np.asarray(arr, dtype=np.float32)
    r = max(int(radius), 0)
    if r == 0:
        return a.astype(np.float32, copy=False)
    valid = np.isfinite(a)
    if not np.any(valid):
        return np.zeros_like(a, dtype=np.float32)
    values = np.where(valid, a, 0.0).astype(np.float32)
    weights = valid.astype(np.float32)
    pad = ((r, r), (r, r))
    values = np.pad(values, pad, mode="edge")
    weights = np.pad(weights, pad, mode="edge")
    sv = np.pad(values, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    sw = np.pad(weights, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    k = 2 * r + 1
    sum_win = sv[k:, k:] - sv[:-k, k:] - sv[k:, :-k] + sv[:-k, :-k]
    cnt_win = sw[k:, k:] - sw[:-k, k:] - sw[k:, :-k] + sw[:-k, :-k]
    out = np.divide(sum_win, np.maximum(cnt_win, 1.0), dtype=np.float32)
    out[cnt_win <= 0.0] = np.nan
    return out.astype(np.float32)


def nan_gaussian_like(arr: np.ndarray, radius: int = 1, passes: int = 2) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    r = max(int(radius), 0)
    p = max(int(passes), 1)
    if r <= 0:
        return out.astype(np.float32, copy=True)
    for _ in range(p):
        out = box_mean(out, radius=r)
    return out.astype(np.float32)


def destripe_axis_banding(
    arr: np.ndarray,
    strength: float = 0.10,
    smooth_radius: int = 6,
    support_frac: float = 0.80,
) -> np.ndarray:
    """Conservative axis-banding suppression.

    Only remove row/column offsets when they are broad and nearly uniform across most of the
    row/column. This avoids the earlier failure mode where one local anomaly changed the whole
    row median/column median and then got painted across the raster.
    """
    a = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(a)
    if not np.any(valid) or min(a.shape) < 8 or float(strength) <= 0.0:
        return a.astype(np.float32, copy=True)

    sr = max(int(smooth_radius), 2)
    base = nan_gaussian_like(a, radius=max(1, sr // 3), passes=2)
    resid = a - base
    resid[~valid] = np.nan
    finite = resid[np.isfinite(resid)]
    if finite.size < 16:
        return a.astype(np.float32, copy=True)
    global_mad = float(np.nanmedian(np.abs(finite - np.nanmedian(finite))))
    if not np.isfinite(global_mad) or global_mad <= 1e-6:
        return a.astype(np.float32, copy=True)

    h, w = a.shape
    row_shift = np.zeros((h,), dtype=np.float32)
    for y in range(h):
        vv = resid[y, :]
        ok = np.isfinite(vv)
        if ok.mean() < support_frac:
            continue
        med = float(np.nanmedian(vv[ok]))
        mad = float(np.nanmedian(np.abs(vv[ok] - med)))
        if abs(med) > 2.5 * global_mad and mad < 0.65 * global_mad:
            row_shift[y] = np.float32(med)
    col_shift = np.zeros((w,), dtype=np.float32)
    for x in range(w):
        vv = resid[:, x]
        ok = np.isfinite(vv)
        if ok.mean() < support_frac:
            continue
        med = float(np.nanmedian(vv[ok]))
        mad = float(np.nanmedian(np.abs(vv[ok] - med)))
        if abs(med) > 2.5 * global_mad and mad < 0.65 * global_mad:
            col_shift[x] = np.float32(med)

    if not np.any(row_shift) and not np.any(col_shift):
        return a.astype(np.float32, copy=True)

    row_shift = np.where(np.abs(row_shift) > 0.0, row_shift - _smooth1d_nan(row_shift, sr), 0.0).astype(np.float32)
    col_shift = np.where(np.abs(col_shift) > 0.0, col_shift - _smooth1d_nan(col_shift, sr), 0.0).astype(np.float32)
    band = row_shift[:, None] + col_shift[None, :]
    out = a - np.float32(strength) * band
    out[~valid] = np.nan
    return out.astype(np.float32)


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(a)
    if not np.any(valid):
        return np.zeros_like(a, dtype=np.float32)
    fill = float(np.nanmedian(a[valid]))
    aa = _nan_to_num(a, fill=fill)
    aa = nan_gaussian_like(aa, radius=1, passes=1)  # keep local denoise only
    gy, gx = np.gradient(aa.astype(np.float32))
    out = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    out[~valid] = np.nan
    return out


def boa_front(arr: np.ndarray, denoise_radius: int = 1, background_radius: int = 3) -> np.ndarray:
    """BOA-inspired front detector without aggressive destriping.

    Keep the operation local: light denoise, broader background subtraction, gradient, then
    a gentle normalize. This avoids painting one anomalous row/column across the field.
    """
    a = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(a)
    if not np.any(valid):
        return np.zeros_like(a, dtype=np.float32)
    sm = nan_gaussian_like(a, radius=max(denoise_radius, 1), passes=1)
    bg = nan_gaussian_like(sm, radius=max(background_radius, 2), passes=1)
    anom = sm - bg
    front = gradient_magnitude(anom)
    front = front.astype(np.float32)
    out = robust_normalize(front, lo_q=20.0, hi_q=95.0, min_span=5e-4)
    return out.astype(np.float32)


def front_persistence(front_stack: Sequence[np.ndarray]) -> np.ndarray:
    if not front_stack:
        raise ValueError("front_stack must not be empty")
    stack = np.stack([np.asarray(x, dtype=np.float32) for x in front_stack], axis=0)
    out = _safe_nanmean_stack(stack)
    return out.astype(np.float32)


def fuse_fronts(
    front_boa_sst: np.ndarray,
    front_boa_logchl: np.ndarray,
    front_ssh: np.ndarray,
    front_persist_3d: np.ndarray | None,
    front_persist_7d: np.ndarray | None,
    weights: Dict[str, float] | None = None,
) -> np.ndarray:
    # SST is the cleanest signal at the current resolutions; CHL/SSH are retained but gently.
    default_w = {"sst": 0.48, "chl": 0.16, "ssh": 0.18, "persist_3d": 0.14, "persist_7d": 0.04}
    w = dict(default_w)
    if weights:
        w.update(weights)
    fields = {
        "sst": np.asarray(front_boa_sst, np.float32),
        "chl": np.asarray(front_boa_logchl, np.float32),
        "ssh": np.asarray(front_ssh, np.float32),
        "persist_3d": np.asarray(front_persist_3d, np.float32) if front_persist_3d is not None else None,
        "persist_7d": np.asarray(front_persist_7d, np.float32) if front_persist_7d is not None else None,
    }
    total = 0.0
    out = np.zeros_like(front_boa_sst, dtype=np.float32)
    for key, arr in fields.items():
        if arr is None:
            continue
        weight = max(float(w.get(key, 0.0)), 0.0)
        if weight <= 0.0:
            continue
        clean = np.asarray(arr, np.float32)
        out += weight * clean
        total += weight
    if total <= 0.0:
        return robust_normalize(front_boa_sst)
    out = out / total
    return robust_normalize(out, lo_q=18.0, hi_q=96.0, min_span=5e-4)


def compute_eke(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu = nan_gaussian_like(np.asarray(u, np.float32), radius=1, passes=1)
    vv = nan_gaussian_like(np.asarray(v, np.float32), radius=1, passes=1)
    uu_a = uu - nan_gaussian_like(uu, radius=2, passes=1)
    vv_a = vv - nan_gaussian_like(vv, radius=2, passes=1)
    out = (0.5 * (uu_a * uu_a + vv_a * vv_a)).astype(np.float32)
    return out.astype(np.float32)


def compute_vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu = nan_gaussian_like(_nan_to_num(u), radius=1, passes=1)
    vv = nan_gaussian_like(_nan_to_num(v), radius=1, passes=1)
    du_dy, du_dx = np.gradient(uu)
    dv_dy, dv_dx = np.gradient(vv)
    out = (dv_dx - du_dy).astype(np.float32)
    return out.astype(np.float32)


def compute_strain(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    uu = nan_gaussian_like(_nan_to_num(u), radius=1, passes=1)
    vv = nan_gaussian_like(_nan_to_num(v), radius=1, passes=1)
    du_dy, du_dx = np.gradient(uu)
    dv_dy, dv_dx = np.gradient(vv)
    s1 = du_dx - dv_dy
    s2 = dv_dx + du_dy
    out = np.sqrt(s1 * s1 + s2 * s2).astype(np.float32)
    return out.astype(np.float32)


def compute_okubo_weiss(vorticity: np.ndarray, strain: np.ndarray) -> np.ndarray:
    vort = np.asarray(vorticity, np.float32)
    st = np.asarray(strain, np.float32)
    out = (st * st - vort * vort).astype(np.float32)
    return out.astype(np.float32)


def detect_eddy_mask(okubo_weiss: np.ndarray, ssh: np.ndarray | None = None) -> np.ndarray:
    ow = np.asarray(okubo_weiss, np.float32)
    thr = float(np.nanpercentile(ow[np.isfinite(ow)], 20)) if np.any(np.isfinite(ow)) else 0.0
    mask = ow < thr
    if ssh is not None:
        amp = robust_normalize(np.abs(np.asarray(ssh, np.float32)))
        mask = mask & (amp > 0.35)
    return mask.astype(np.uint8)


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    inf = np.float32(1e9)
    dist = np.full((h, w), inf, dtype=np.float32)
    dist[m] = 0.0
    root2 = np.float32(math.sqrt(2.0))
    for y in range(h):
        for x in range(w):
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + root2)
                if x + 1 < w:
                    best = min(best, dist[y - 1, x + 1] + root2)
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            dist[y, x] = best
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            best = dist[y, x]
            if y + 1 < h:
                best = min(best, dist[y + 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + root2)
                if x + 1 < w:
                    best = min(best, dist[y + 1, x + 1] + root2)
            if x + 1 < w:
                best = min(best, dist[y, x + 1] + 1.0)
            dist[y, x] = best
    finite = np.isfinite(dist)
    if np.any(finite):
        dist[~finite] = np.nanmax(dist[finite])
    else:
        dist[:] = 0.0
    return dist.astype(np.float32)


def compute_eddy_edge_distance(mask: np.ndarray) -> np.ndarray:
    return robust_normalize(distance_to_mask(mask > 0))


def rolling_mean(layers_by_tid: Dict[str, Dict[str, np.ndarray]], ordered_time_ids: Sequence[str], current_tid: str, key: str, window_steps: int) -> np.ndarray:
    idx = ordered_time_ids.index(current_tid)
    lo = max(0, idx - max(int(window_steps), 1) + 1)
    tids = ordered_time_ids[lo:idx + 1]
    stack = [np.asarray(layers_by_tid[tid][key], np.float32) for tid in tids if key in layers_by_tid[tid]]
    return _safe_nanmean_stack(np.stack(stack, axis=0)).astype(np.float32) if stack else np.asarray(layers_by_tid[current_tid][key], np.float32)


def anomaly(arr: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return robust_normalize(np.asarray(arr, np.float32) - np.asarray(baseline, np.float32))


def score_mld(mld: np.ndarray) -> np.ndarray:
    x = np.asarray(mld, np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    med = float(np.nanmedian(x))
    sig = float(max(np.nanstd(x), 5.0))
    out = np.exp(-0.5 * ((x - med) / sig) ** 2)
    return robust_normalize(out)


def score_o2(o2: np.ndarray) -> np.ndarray:
    return robust_normalize(np.asarray(o2, np.float32))


def score_sss(sss: np.ndarray) -> np.ndarray:
    x = np.asarray(sss, np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    mu = float(np.nanmedian(x))
    sig = float(max(np.nanstd(x), 0.25))
    out = np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return robust_normalize(out)


def vertical_access(mld: np.ndarray, o2: np.ndarray, sss: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    smld = score_mld(mld)
    so2 = score_o2(o2)
    ssss = score_sss(sss)
    out = np.clip(0.35 * smld + 0.45 * so2 + 0.20 * ssss, 0.0, 1.0).astype(np.float32)
    return out, {"mld_score": smld, "o2_score": so2, "sss_score": ssss}


def wind_speed_dir(u10: np.ndarray, v10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u10, np.float32)
    v = np.asarray(v10, np.float32)
    speed = np.sqrt(u * u + v * v).astype(np.float32)
    direction = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    return speed.astype(np.float32), direction.astype(np.float32)


def wind_penalty(speed: np.ndarray, soft_min: float = 5.0, soft_max: float = 12.0) -> np.ndarray:
    ws = np.asarray(speed, np.float32)
    mid = 0.5 * (soft_min + soft_max)
    scale = max((soft_max - soft_min) / 4.0, 0.75)
    out = 1.0 / (1.0 + np.exp((ws - mid) / scale))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def thermocline_proxy(mld: np.ndarray) -> np.ndarray:
    x = np.asarray(mld, np.float32)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.zeros_like(x, dtype=np.float32)
    med = float(np.nanmedian(x))
    sig = float(max(np.nanstd(x), 5.0))
    out = np.exp(-0.5 * ((x - med) / sig) ** 2)
    return robust_normalize(out)
