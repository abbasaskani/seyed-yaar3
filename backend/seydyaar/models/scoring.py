from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .ocean_features import gradient_magnitude, robust_normalize


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((np.asarray(x, dtype=np.float32) - np.float32(mu)) / np.float32(sigma)) ** 2).astype(np.float32)


def _logistic_lower(x: np.ndarray, z0: float, k: float) -> np.ndarray:
    k = max(float(k), 1e-6)
    xx = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-(xx - np.float32(z0)) / np.float32(k)))).astype(np.float32)


def _clean_prob_field(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def score_temp_c(sst_c: np.ndarray, opt_c: float, sigma_c: float) -> np.ndarray:
    return _clean_prob_field(_gauss(sst_c, opt_c, sigma_c))


def score_chl_mg_m3(chl: np.ndarray, opt_mg_m3: float, sigma_log10: float) -> np.ndarray:
    chl = np.clip(np.asarray(chl, dtype=np.float32), 1e-6, None)
    return _clean_prob_field(_gauss(np.log10(chl), np.log10(max(float(opt_mg_m3), 1e-6)), sigma_log10))


def score_current_m_s(spd: np.ndarray, opt_m_s: float, sigma_m_s: float) -> np.ndarray:
    return _clean_prob_field(_gauss(spd, opt_m_s, sigma_m_s))


def score_waves_hs(hs_m: np.ndarray, soft_max_m: float = 1.5, softness: float = 0.35) -> np.ndarray:
    soft = max(float(softness), 1e-6)
    hs = np.asarray(hs_m, dtype=np.float32)
    return _clean_prob_field(1.0 / (1.0 + np.exp((hs - np.float32(soft_max_m)) / np.float32(soft))))


def score_mld(mld: np.ndarray, opt_m: float, sigma_m: float) -> np.ndarray:
    return _clean_prob_field(_gauss(mld, opt_m, sigma_m))


def score_sss(sss: np.ndarray, opt_psu: float, sigma_psu: float) -> np.ndarray:
    return _clean_prob_field(_gauss(sss, opt_psu, sigma_psu))


def score_o2(o2: np.ndarray, opt_mmol_m3: float, sigma_mmol_m3: float, min_mmol_m3: float, min_softness_mmol_m3: float) -> np.ndarray:
    upper = _gauss(o2, opt_mmol_m3, sigma_mmol_m3)
    lower = _logistic_lower(o2, min_mmol_m3, min_softness_mmol_m3)
    return _clean_prob_field(upper * lower)


def front_score(temp_front: np.ndarray, chl_front: np.ndarray, ssh_front: np.ndarray,
                w_temp: float = 0.4, w_chl: float = 0.4, w_ssh: float = 0.2) -> np.ndarray:
    s = np.float32(w_temp) * np.asarray(temp_front, np.float32) + np.float32(w_chl) * np.asarray(chl_front, np.float32) + np.float32(w_ssh) * np.asarray(ssh_front, np.float32)
    return robust_normalize(s, lo_q=15.0, hi_q=95.0, min_span=5e-4)


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
    mld_m: np.ndarray | None = None
    o2_mmol_m3: np.ndarray | None = None
    sss_psu: np.ndarray | None = None


def habitat_scoring(inputs: HabitatInputs, priors: Dict, weights: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    w = {k: max(float(v), 0.0) for k, v in dict(weights).items()}
    total = sum(w.values())
    if total <= 0:
        w = {"temp": 1.0}
        total = 1.0
    for k in list(w.keys()):
        w[k] = w[k] / total

    s_temp = score_temp_c(inputs.sst_c, priors["sst_opt_c"], priors["sst_sigma_c"])
    s_chl = score_chl_mg_m3(inputs.chl_mg_m3, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"])
    s_cur = score_current_m_s(inputs.current_m_s, priors["current_opt_m_s"], priors["current_sigma_m_s"])
    s_waves = score_waves_hs(inputs.waves_hs_m, priors.get("waves_hs_soft_max_m", 1.5))

    tf = cf = sf = None
    if w.get("front", 0.0) > 0.0:
        if inputs.front_fused is not None:
            s_front = np.asarray(inputs.front_fused, dtype=np.float32)
        else:
            tf = gradient_magnitude(inputs.sst_c, denoise_radius=0)
            cf = gradient_magnitude(np.log10(np.clip(inputs.chl_mg_m3, 1e-6, None)), denoise_radius=0)
            sf = gradient_magnitude(inputs.ssh_m, denoise_radius=0)
            fw = priors.get("front_weights", {"temp": 0.4, "chl": 0.4, "ssh": 0.2})
            s_front = front_score(tf, cf, sf, fw.get("temp", 0.4), fw.get("chl", 0.4), fw.get("ssh", 0.2))
        s_front = np.power(np.clip(s_front, 0.0, 1.0), 1.08).astype(np.float32)
    else:
        s_front = np.zeros_like(s_temp, dtype=np.float32)

    zero = np.zeros_like(s_temp, dtype=np.float32)
    s_eke = robust_normalize(inputs.eke, lo_q=10.0, hi_q=92.0, min_span=5e-4) if (inputs.eke is not None and w.get("eke", 0.0) > 0.0) else zero
    s_okubo = robust_normalize(-np.asarray(inputs.okubo_weiss, dtype=np.float32), lo_q=10.0, hi_q=92.0, min_span=5e-4) if (inputs.okubo_weiss is not None and w.get("okubo_weiss", 0.0) > 0.0) else zero
    s_eddy_edge = _clean_prob_field(1.0 - np.asarray(inputs.eddy_edge_distance, dtype=np.float32)) if (inputs.eddy_edge_distance is not None and w.get("eddy_edge", 0.0) > 0.0) else zero
    s_mld = score_mld(inputs.mld_m, priors.get("mld_opt_m", 80.0), priors.get("mld_sigma_m", 25.0)) if (inputs.mld_m is not None and w.get("mld", 0.0) > 0.0) else zero
    s_o2 = score_o2(inputs.o2_mmol_m3, priors.get("o2_opt_mmol_m3", 200.0), priors.get("o2_sigma_mmol_m3", 5.0), priors.get("o2_min_mmol_m3", 145.0), priors.get("o2_min_softness_mmol_m3", 8.0)) if (inputs.o2_mmol_m3 is not None and w.get("o2", 0.0) > 0.0) else zero
    s_sss = score_sss(inputs.sss_psu, priors.get("sss_opt_psu", 35.0), priors.get("sss_sigma_psu", 0.35)) if (inputs.sss_psu is not None and w.get("sss", 0.0) > 0.0) else zero
    if inputs.vertical_access is not None and w.get("vertical", 0.0) > 0.0:
        s_vertical = _clean_prob_field(np.asarray(inputs.vertical_access, dtype=np.float32))
    else:
        # derive a vertical score if the combined proxy was not provided
        if (inputs.mld_m is not None) and (inputs.o2_mmol_m3 is not None) and (inputs.sss_psu is not None):
            s_vertical = _clean_prob_field(0.35 * s_mld + 0.45 * s_o2 + 0.20 * s_sss)
        else:
            s_vertical = zero
    s_chl_3d = score_chl_mg_m3(inputs.chl_3d_mean, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"]) if (inputs.chl_3d_mean is not None and w.get("chl_3d", 0.0) > 0.0) else zero
    s_chl_7d = score_chl_mg_m3(inputs.chl_7d_mean, priors["chl_opt_mg_m3"], priors["chl_sigma_log10"]) if (inputs.chl_7d_mean is not None and w.get("chl_7d", 0.0) > 0.0) else zero
    s_chl_anom = _clean_prob_field(np.asarray(inputs.chl_anom, dtype=np.float32)) if (inputs.chl_anom is not None and w.get("chl_anom", 0.0) > 0.0) else zero
    s_npp_anom = _clean_prob_field(np.asarray(inputs.npp_anom, dtype=np.float32)) if (inputs.npp_anom is not None and w.get("npp_anom", 0.0) > 0.0) else zero
    if inputs.thermocline_proxy is not None and w.get("thermocline", 0.0) > 0.0:
        # thermocline_proxy is an indirect field; score it against literature-informed depth priors where possible
        s_thermocline = score_mld(inputs.thermocline_proxy, priors.get("thermocline_opt_m", priors.get("mld_opt_m", 80.0)), priors.get("thermocline_sigma_m", priors.get("mld_sigma_m", 25.0)))
    else:
        s_thermocline = zero

    core = (
        w.get("temp", 0.0) * s_temp +
        w.get("chl", 0.0) * s_chl +
        w.get("current", 0.0) * s_cur +
        w.get("vertical", 0.0) * s_vertical +
        w.get("mld", 0.0) * s_mld +
        w.get("o2", 0.0) * s_o2 +
        w.get("sss", 0.0) * s_sss +
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
    # Keep mesoscale refinement meaningful but not dominant; this constant remains heuristic.
    phab = _clean_prob_field(core + np.float32(0.70) * meso)

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
        "score_mld": s_mld,
        "score_o2": s_o2,
        "score_sss": s_sss,
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
