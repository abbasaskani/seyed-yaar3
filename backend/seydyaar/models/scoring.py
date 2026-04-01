from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .ocean_features import gradient_magnitude, robust_normalize


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def score_temp_c(sst_c: np.ndarray, opt_c: float, sigma_c: float) -> np.ndarray:
    return _gauss(sst_c, opt_c, sigma_c)


def score_chl_mg_m3(chl: np.ndarray, opt_mg_m3: float, sigma_log10: float) -> np.ndarray:
    chl = np.clip(chl, 1e-6, None)
    return _gauss(np.log10(chl), np.log10(opt_mg_m3), sigma_log10)


def score_current_m_s(spd: np.ndarray, opt_m_s: float, sigma_m_s: float) -> np.ndarray:
    return _gauss(spd, opt_m_s, sigma_m_s)


def score_waves_hs(hs_m: np.ndarray, soft_max_m: float = 1.5, softness: float = 0.35) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((hs_m - soft_max_m) / max(softness, 1e-6)))


def front_score(temp_front: np.ndarray, chl_front: np.ndarray, ssh_front: np.ndarray,
                w_temp: float = 0.5, w_chl: float = 0.25, w_ssh: float = 0.25) -> np.ndarray:
    s = w_temp * temp_front + w_chl * chl_front + w_ssh * ssh_front
    # Use robust normalization with a slightly tighter band and soften the top end
    # so front artifacts do not dominate habitat on compact AOIs.
    s = robust_normalize(s, lo_q=10.0, hi_q=90.0)
    return np.sqrt(np.clip(s, 0.0, 1.0)).astype(np.float32)


@dataclass
class HabitatInputs:
    sst_c: np.ndarray
    chl_mg_m3: np.ndarray
    current_m_s: np.ndarray
    waves_hs_m: np.ndarray
    ssh_m: np.ndarray
    front_fused: np.ndarray | None = None
    eke: np.ndarray | None = None
    vorticity: np.ndarray | None = None
    strain: np.ndarray | None = None
    okubo_weiss: np.ndarray | None = None
    eddy_edge_distance: np.ndarray | None = None
    vertical_access: np.ndarray | None = None
    chl_3d_mean: np.ndarray | None = None
    chl_7d_mean: np.ndarray | None = None
    chl_anom: np.ndarray | None = None
    npp_anom: np.ndarray | None = None
    thermocline_proxy: np.ndarray | None = None


def habitat_scoring(inputs: HabitatInputs, priors: Dict, weights: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    s_temp = score_temp_c(inputs.sst_c, priors["sst_opt_c"], priors["sst_sigma_c"])
    s_chl = score_chl_mg_m3(inputs.chl_mg_m3, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"])
    s_cur = score_current_m_s(inputs.current_m_s, priors["current_opt_m_s"], priors["current_sigma_m_s"])
    s_waves = score_waves_hs(inputs.waves_hs_m, priors.get("waves_hs_soft_max_m", 1.5))

    tf = cf = sf = None
    if inputs.front_fused is not None:
        s_front = np.asarray(inputs.front_fused, dtype=np.float32)
    else:
        tf = gradient_magnitude(inputs.sst_c)
        cf = gradient_magnitude(np.log10(np.clip(inputs.chl_mg_m3, 1e-6, None)))
        sf = gradient_magnitude(inputs.ssh_m)
        fw = priors.get("front_weights", {"temp": 0.5, "chl": 0.25, "ssh": 0.25})
        s_front = front_score(tf, cf, sf, fw.get("temp", 0.5), fw.get("chl", 0.25), fw.get("ssh", 0.25))

    zero = np.zeros_like(s_front, dtype=np.float32)
    s_eke = robust_normalize(inputs.eke) if inputs.eke is not None else zero
    s_okubo = robust_normalize(-np.asarray(inputs.okubo_weiss, dtype=np.float32)) if inputs.okubo_weiss is not None else zero
    s_eddy_edge = (1.0 - np.asarray(inputs.eddy_edge_distance, dtype=np.float32)) if inputs.eddy_edge_distance is not None else zero
    s_vertical = np.asarray(inputs.vertical_access, dtype=np.float32) if inputs.vertical_access is not None else zero
    s_chl_3d = score_chl_mg_m3(inputs.chl_3d_mean, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"]) if inputs.chl_3d_mean is not None else zero
    s_chl_7d = score_chl_mg_m3(inputs.chl_7d_mean, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"]) if inputs.chl_7d_mean is not None else zero
    s_chl_anom = np.asarray(inputs.chl_anom, dtype=np.float32) if inputs.chl_anom is not None else zero
    s_npp_anom = np.asarray(inputs.npp_anom, dtype=np.float32) if inputs.npp_anom is not None else zero
    s_thermocline = np.asarray(inputs.thermocline_proxy, dtype=np.float32) if inputs.thermocline_proxy is not None else zero

    w = dict(weights)
    total = sum(max(float(v), 0.0) for v in w.values())
    if total <= 0:
        w = {"temp": 1.0}
        total = 1.0
    for k in list(w.keys()):
        w[k] = max(float(w[k]), 0.0) / total

    # Slightly damp the front contribution in the final blend; front remains
    # important but should not create visually dominant striping on its own.
    front_blend = 0.75 * s_front + 0.25 * robust_normalize(s_front, lo_q=20.0, hi_q=80.0)

    phab = (
        w.get("temp", 0.0) * s_temp +
        w.get("chl", 0.0) * s_chl +
        w.get("front", 0.0) * front_blend +
        w.get("current", 0.0) * s_cur +
        w.get("eke", 0.0) * s_eke +
        w.get("okubo_weiss", 0.0) * s_okubo +
        w.get("eddy_edge", 0.0) * s_eddy_edge +
        w.get("vertical", 0.0) * s_vertical +
        w.get("chl_3d", 0.0) * s_chl_3d +
        w.get("chl_7d", 0.0) * s_chl_7d +
        w.get("chl_anom", 0.0) * s_chl_anom +
        w.get("npp_anom", 0.0) * s_npp_anom +
        w.get("thermocline", 0.0) * s_thermocline
    )
    phab = np.clip(phab, 0.0, 1.0).astype(np.float32)

    comps = {
        "score_temp": s_temp,
        "score_chl": s_chl,
        "score_front": s_front,
        "score_current": s_cur,
        "score_waves": s_waves,
        "score_eke": s_eke,
        "score_okubo_weiss": s_okubo,
        "score_eddy_edge": s_eddy_edge,
        "score_vertical": s_vertical,
        "score_chl_3d": s_chl_3d,
        "score_chl_7d": s_chl_7d,
        "score_chl_anom": s_chl_anom,
        "score_npp_anom": s_npp_anom,
        "score_thermocline": s_thermocline,
    }
    if tf is not None:
        comps["temp_front"] = tf
    if cf is not None:
        comps["chl_front"] = cf
    if sf is not None:
        comps["ssh_front"] = sf
    return phab, comps
