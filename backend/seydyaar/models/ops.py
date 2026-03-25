from __future__ import annotations

from typing import Dict
import numpy as np

from .scoring import score_current_m_s, score_waves_hs
from .ocean_features import wind_penalty


def ops_feasibility(
    current_m_s: np.ndarray,
    waves_hs_m: np.ndarray,
    priors: Dict,
    gear_depth_m: float = 10.0,
    wind_speed_m_s: np.ndarray | None = None,
) -> np.ndarray:
    """Operational feasibility Pops (0..1).

    Lightweight and stable: waves + currents + optional simple wind penalty.
    Depth only changes relative weighting mildly.
    """
    soft_max = float(priors.get("waves_hs_soft_max_m", 1.5))
    s_w = score_waves_hs(waves_hs_m, soft_max_m=soft_max)

    opt = float(priors.get("current_opt_m_s", 0.4))
    sig = float(priors.get("current_sigma_m_s", 0.25))
    s_c = score_current_m_s(current_m_s, opt_m_s=opt, sigma_m_s=sig)

    d = float(gear_depth_m)
    w_waves = 0.52 + (10.0 - d) * 0.01
    w_curr = 0.48 + (d - 10.0) * 0.01
    if wind_speed_m_s is None:
        w_wind = 0.0
    else:
        w_wind = 0.18
        w_waves -= 0.09
        w_curr -= 0.09
    w_waves = float(np.clip(w_waves, 0.35, 0.70))
    w_curr = float(np.clip(w_curr, 0.25, 0.60))
    total = w_waves + w_curr + w_wind
    w_waves /= total
    w_curr /= total
    w_wind /= total

    pops = w_waves * s_w + w_curr * s_c
    if wind_speed_m_s is not None:
        ws = np.asarray(wind_speed_m_s, dtype=np.float32)
        s_wind = wind_penalty(ws, soft_min=float(priors.get("wind_soft_min_m_s", 5.0)), soft_max=float(priors.get("wind_soft_max_m_s", 12.0)))
        pops = pops + w_wind * s_wind
    return np.clip(pops, 0.0, 1.0).astype(np.float32)
