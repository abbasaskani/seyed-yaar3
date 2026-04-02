from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .ocean_features import (
    box_mean,
    destripe_axis_banding,
    gradient_magnitude,
    nan_gaussian_like,
    robust_normalize,
)


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _clean_prob_field(a: np.ndarray, smooth_radius: int = 1, stripe_strength: float = 0.0) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32)
    if smooth_radius > 0:
        out = nan_gaussian_like(out, radius=int(smooth_radius), passes=1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


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
                w_temp: float = 0.6, w_chl: float = 0.15, w_ssh: float = 0.25) -> np.ndarray:
    s = w_temp * temp_front + w_chl * chl_front + w_ssh * ssh_front
    s = _clean_prob_field(s, smooth_radius=1, stripe_strength=0.0)
    return robust_normalize(s, lo_q=15.0, hi_q=92.0, min_span=5e-4)


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
    w = {k: max(float(v), 0.0) for k, v in dict(weights).items()}
    total = sum(w.values())
    if total <= 0:
        w = {"temp": 1.0}
        total = 1.0
    for k in list(w.keys()):
        w[k] = w[k] / total

    s_temp = _clean_prob_field(score_temp_c(inputs.sst_c, priors["sst_opt_c"], priors["sst_sigma_c"]), smooth_radius=1, stripe_strength=0.0)
    s_chl = _clean_prob_field(score_chl_mg_m3(inputs.chl_mg_m3, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"]), smooth_radius=1, stripe_strength=0.0)
    s_cur = _clean_prob_field(score_current_m_s(inputs.current_m_s, priors["current_opt_m_s"], priors["current_sigma_m_s"]), smooth_radius=1, stripe_strength=0.0)
    s_waves = _clean_prob_field(score_waves_hs(inputs.waves_hs_m, priors.get("waves_hs_soft_max_m", 1.5)), smooth_radius=1, stripe_strength=0.0)

    tf = cf = sf = None
    if w.get("front", 0.0) > 0.0:
        if inputs.front_fused is not None:
            s_front = np.asarray(inputs.front_fused, dtype=np.float32)
        else:
            tf = gradient_magnitude(inputs.sst_c)
            cf = gradient_magnitude(np.log10(np.clip(inputs.chl_mg_m3, 1e-6, None)))
            sf = gradient_magnitude(inputs.ssh_m)
            fw = priors.get("front_weights", {"temp": 0.6, "chl": 0.15, "ssh": 0.25})
            s_front = front_score(tf, cf, sf, fw.get("temp", 0.6), fw.get("chl", 0.15), fw.get("ssh", 0.25))
        s_front = _clean_prob_field(s_front, smooth_radius=1, stripe_strength=0.0)
        s_front = np.power(np.clip(s_front, 0.0, 1.0), 1.20).astype(np.float32)
    else:
        s_front = np.zeros_like(s_temp, dtype=np.float32)

    zero = np.zeros_like(s_temp, dtype=np.float32)
    s_eke = _clean_prob_field(robust_normalize(inputs.eke, lo_q=10.0, hi_q=90.0, min_span=5e-4), smooth_radius=1, stripe_strength=0.0) if (inputs.eke is not None and w.get("eke", 0.0) > 0.0) else zero
    s_okubo = _clean_prob_field(robust_normalize(-np.asarray(inputs.okubo_weiss, dtype=np.float32), lo_q=10.0, hi_q=90.0, min_span=5e-4), smooth_radius=1, stripe_strength=0.0) if (inputs.okubo_weiss is not None and w.get("okubo_weiss", 0.0) > 0.0) else zero
    s_eddy_edge = _clean_prob_field((1.0 - np.asarray(inputs.eddy_edge_distance, dtype=np.float32)), smooth_radius=1, stripe_strength=0.0) if (inputs.eddy_edge_distance is not None and w.get("eddy_edge", 0.0) > 0.0) else zero
    s_vertical = _clean_prob_field(np.asarray(inputs.vertical_access, dtype=np.float32), smooth_radius=1, stripe_strength=0.0) if (inputs.vertical_access is not None and w.get("vertical", 0.0) > 0.0) else zero
    s_chl_3d = _clean_prob_field(score_chl_mg_m3(inputs.chl_3d_mean, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"]), smooth_radius=1, stripe_strength=0.0) if (inputs.chl_3d_mean is not None and w.get("chl_3d", 0.0) > 0.0) else zero
    s_chl_7d = _clean_prob_field(score_chl_mg_m3(inputs.chl_7d_mean, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"]), smooth_radius=1, stripe_strength=0.0) if (inputs.chl_7d_mean is not None and w.get("chl_7d", 0.0) > 0.0) else zero
    s_chl_anom = _clean_prob_field(np.asarray(inputs.chl_anom, dtype=np.float32), smooth_radius=1, stripe_strength=0.0) if (inputs.chl_anom is not None and w.get("chl_anom", 0.0) > 0.0) else zero
    s_npp_anom = _clean_prob_field(np.asarray(inputs.npp_anom, dtype=np.float32), smooth_radius=1, stripe_strength=0.0) if (inputs.npp_anom is not None and w.get("npp_anom", 0.0) > 0.0) else zero
    s_thermocline = _clean_prob_field(np.asarray(inputs.thermocline_proxy, dtype=np.float32), smooth_radius=1, stripe_strength=0.0) if (inputs.thermocline_proxy is not None and w.get("thermocline", 0.0) > 0.0) else zero

    # Keep core habitat dominant; let mesoscale fields refine rather than dominate.
    core = (
        w.get("temp", 0.0) * s_temp +
        w.get("chl", 0.0) * s_chl +
        w.get("current", 0.0) * s_cur +
        w.get("vertical", 0.0) * s_vertical +
        w.get("chl_3d", 0.0) * s_chl_3d +
        w.get("chl_7d", 0.0) * s_chl_7d +
        w.get("thermocline", 0.0) * s_thermocline
    )
    meso = (
        w.get("front", 0.0) * s_front +
        w.get("eke", 0.0) * s_eke +
        w.get("okubo_weiss", 0.0) * s_okubo +
        w.get("eddy_edge", 0.0) * s_eddy_edge +
        w.get("chl_anom", 0.0) * s_chl_anom +
        w.get("npp_anom", 0.0) * s_npp_anom
    )
    phab = core + np.float32(0.72) * meso
    phab = _clean_prob_field(phab, smooth_radius=1, stripe_strength=0.0)
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
