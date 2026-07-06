"""Metrics-only ODE simulator for ODE-back functional validation (no full trajectories returned)."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from multi_pathogen_simulator import (
    N_STRAINS,
    NUMBA_SIMULATION_ENABLED,
    PAPER_FIGURE_PROFILE,
    _simulate_forward_euler_numba,
    simulate_paper_case,
)

TERMINAL_WINDOW_H = 12.0

M_P_AUC = 0
M_LR1 = 1
M_LR5 = 5
M_MEAN_LR = 6
M_TOTAL_DOSAGE = 7
M_DOSE_COUNT = 8
M_FINAL_TOTAL_PATHOGEN = 9
M_TERMINAL_TOTAL_PATHOGEN = 10
M_LOG10_TERMINAL = 11
M_LOG10_FINAL = 12
N_METRICS = 13

METRIC_NAMES = (
    "P_AUC",
    "LR1",
    "LR2",
    "LR3",
    "LR4",
    "LR5",
    "mean_LR",
    "total_dosage",
    "dose_count",
    "final_total_pathogen",
    "terminal_total_pathogen",
    "log10_terminal_total_pathogen",
    "log10_final_total_pathogen",
)


def metrics_vector_to_dict(metrics: np.ndarray) -> Dict[str, float]:
    metrics = np.asarray(metrics, dtype=float).ravel()
    return {name: float(metrics[idx]) for idx, name in enumerate(METRIC_NAMES)}


def _log10_safe(value: float) -> float:
    return float(np.log10(max(value, 1.0)))


def _profile_euler_args(profile) -> Tuple:
    return (
        profile.K_pathogen,
        profile.eta_pathogen,
        profile.alpha,
        profile.beta,
        profile.probiotic_model == "two_compartment",
        profile.K_P,
        profile.k_P,
        profile.gamma_P,
        profile.rho_P,
        profile.eta_P,
        profile.mu_P,
        profile.P0,
        profile.lambda_amp,
        profile.dt_detect,
        profile.t_end,
        profile.dt,
    )


def _metrics_from_histories(
    B0: np.ndarray,
    B_S_hist: np.ndarray,
    B_T_hist: np.ndarray,
    P_hist: np.ndarray,
    dose_count: int,
    u_max: float,
    K_P: float,
    t_end: float,
    steps: int,
    *,
    C_hist: np.ndarray | None = None,
) -> np.ndarray:
    """Compute outcome metrics from forward-Euler histories (no trajectory export)."""
    times = np.linspace(0.0, t_end, steps)
    B_total = B_S_hist + B_T_hist
    terminal_mask = times >= (t_end - TERMINAL_WINDOW_H - 1e-12)
    B_terminal = B_total[terminal_mask]
    LR = np.log10(B0 / np.clip(np.median(B_terminal, axis=0), 1.0, None))
    terminal_total = float(np.median(B_terminal.sum(axis=1)))
    final_total = float(B_total[-1].sum())
    P_AUC = float(np.trapezoid(P_hist / K_P, times) / t_end)

    out = np.empty(N_METRICS)
    out[M_P_AUC] = P_AUC
    for i in range(N_STRAINS):
        out[M_LR1 + i] = LR[i]
    out[M_MEAN_LR] = float(np.mean(LR))
    out[M_TOTAL_DOSAGE] = float(dose_count) * float(u_max)
    out[M_DOSE_COUNT] = float(dose_count)
    out[M_FINAL_TOTAL_PATHOGEN] = final_total
    out[M_TERMINAL_TOTAL_PATHOGEN] = terminal_total
    out[M_LOG10_TERMINAL] = _log10_safe(terminal_total)
    out[M_LOG10_FINAL] = _log10_safe(final_total)
    return out


def _metrics_from_legacy_sim(B0: np.ndarray, res) -> np.ndarray:
    terminal_mask = res.times >= (res.times[-1] - TERMINAL_WINDOW_H - 1e-12)
    terminal_total = float(np.median(res.B_total[terminal_mask].sum(axis=1)))
    final_total = float(res.B_total[-1].sum())
    out = np.empty(N_METRICS)
    out[M_P_AUC] = float(res.P_AUC)
    for i in range(N_STRAINS):
        out[M_LR1 + i] = float(res.LR[i])
    out[M_MEAN_LR] = float(np.mean(res.LR))
    out[M_TOTAL_DOSAGE] = float(res.total_dosage)
    out[M_DOSE_COUNT] = float(res.dose_count)
    out[M_FINAL_TOTAL_PATHOGEN] = final_total
    out[M_TERMINAL_TOTAL_PATHOGEN] = terminal_total
    out[M_LOG10_TERMINAL] = _log10_safe(terminal_total)
    out[M_LOG10_FINAL] = _log10_safe(final_total)
    return out


def _log10_clip(values: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(values, dtype=float), 1.0, None))


def trajectory_feature_vector(
    B_S_hist: np.ndarray,
    B_T_hist: np.ndarray,
    P_hist: np.ndarray,
    C_hist: np.ndarray,
) -> np.ndarray:
    """Flatten log-scaled ODE trajectories for pred-vs-ref R² (AMP, probiotic, pathogens)."""
    B_total = np.asarray(B_S_hist, dtype=float) + np.asarray(B_T_hist, dtype=float)
    P = np.asarray(P_hist, dtype=float)
    C = np.asarray(C_hist, dtype=float)
    parts = [
        C,
        _log10_clip(P),
        _log10_clip(B_total.sum(axis=1)),
    ]
    for i in range(N_STRAINS):
        parts.append(_log10_clip(B_total[:, i]))
    return np.concatenate(parts, axis=0)


def _simulate_histories_fast(
    B0: np.ndarray,
    k_arr: np.ndarray,
    gamma_arr: np.ndarray,
    rho_arr: np.ndarray,
    mu_arr: np.ndarray,
    u_max: float,
    T_thr: np.ndarray,
    *,
    profile=PAPER_FIGURE_PROFILE,
    backend: str = "numba",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Run forward Euler once; return histories used for metrics and trajectory R²."""
    B0 = np.asarray(B0, dtype=float).reshape(N_STRAINS)
    k_arr = np.asarray(k_arr, dtype=float).reshape(N_STRAINS)
    gamma_arr = np.asarray(gamma_arr, dtype=float).reshape(N_STRAINS)
    rho_arr = np.asarray(rho_arr, dtype=float).reshape(N_STRAINS)
    mu_arr = np.asarray(mu_arr, dtype=float).reshape(N_STRAINS)
    T_thr = np.asarray(T_thr, dtype=float).reshape(N_STRAINS)
    gamma_T = gamma_arr * (1.0 - rho_arr)

    if backend == "python":
        res = simulate_paper_case(B0, k_arr, gamma_arr, rho_arr, mu_arr, float(u_max), T_thr)
        steps = int(res.times.size)
        P_total = np.asarray(res.P_S, dtype=float) + np.asarray(res.P_R, dtype=float)
        return res.B_S, res.B_R, P_total, res.C, int(res.dose_count), steps

    if backend == "numba" and NUMBA_SIMULATION_ENABLED:
        (
            B_S_hist,
            B_T_hist,
            P_hist,
            _P_S_hist,
            _P_T_hist,
            C_hist,
            _dose_times_buf,
            dose_count,
            steps,
        ) = _simulate_forward_euler_numba(
            B0,
            k_arr,
            gamma_arr,
            gamma_T,
            mu_arr,
            float(u_max),
            T_thr,
            *_profile_euler_args(profile),
        )
        return B_S_hist, B_T_hist, P_hist, C_hist, int(dose_count), int(steps)

    res = simulate_paper_case(B0, k_arr, gamma_arr, rho_arr, mu_arr, float(u_max), T_thr)
    steps = int(res.times.size)
    P_total = np.asarray(res.P_S, dtype=float) + np.asarray(res.P_R, dtype=float)
    return res.B_S, res.B_R, P_total, res.C, int(res.dose_count), steps


def simulate_case_metrics_and_trajectory_fast(
    B0: np.ndarray,
    k_arr: np.ndarray,
    gamma_arr: np.ndarray,
    rho_arr: np.ndarray,
    mu_arr: np.ndarray,
    u_max: float,
    T_thr: np.ndarray,
    *,
    profile=PAPER_FIGURE_PROFILE,
    backend: str = "numba",
) -> Tuple[np.ndarray, np.ndarray]:
    """Single ODE run: outcome metrics vector + flattened trajectory features."""
    B_S_hist, B_T_hist, P_hist, C_hist, dose_count, steps = _simulate_histories_fast(
        B0, k_arr, gamma_arr, rho_arr, mu_arr, u_max, T_thr, profile=profile, backend=backend
    )
    metrics = _metrics_from_histories(
        B0,
        B_S_hist,
        B_T_hist,
        P_hist,
        dose_count,
        float(u_max),
        float(profile.K_P),
        float(profile.t_end),
        int(steps),
        C_hist=C_hist,
    )
    traj = trajectory_feature_vector(B_S_hist, B_T_hist, P_hist, C_hist)
    return metrics, traj


def simulate_case_metrics_fast(
    B0: np.ndarray,
    k_arr: np.ndarray,
    gamma_arr: np.ndarray,
    rho_arr: np.ndarray,
    mu_arr: np.ndarray,
    u_max: float,
    T_thr: np.ndarray,
    *,
    profile=PAPER_FIGURE_PROFILE,
    backend: str = "numba",
) -> np.ndarray:
    """Return metrics vector (length N_METRICS); caller does not receive full trajectories."""
    metrics, _ = simulate_case_metrics_and_trajectory_fast(
        B0,
        k_arr,
        gamma_arr,
        rho_arr,
        mu_arr,
        u_max,
        T_thr,
        profile=profile,
        backend=backend,
    )
    return metrics


def validate_fast_backend(
    n_samples: int = 10,
    seed: int = 42,
    rtol: float = 1e-5,
    atol: float = 1e-4,
) -> Tuple[bool, list]:
    """Compare numba metrics path vs legacy simulate_paper_case on random cases."""
    rng = np.random.default_rng(seed)
    profile = PAPER_FIGURE_PROFILE
    warnings: list = []
    ok = True

    for _ in range(n_samples):
        B0 = profile.B0_rep * rng.uniform(0.8, 1.2, N_STRAINS)
        k_arr = profile.k_rep * rng.uniform(0.9, 1.1, N_STRAINS)
        gamma_arr = profile.gamma_s_rep * rng.uniform(0.9, 1.1, N_STRAINS)
        rho_arr = np.clip(profile.rho_rep + rng.uniform(-0.05, 0.05, N_STRAINS), 0.0, 0.99)
        mu_arr = profile.mu_rep * rng.uniform(0.9, 1.1, N_STRAINS)
        T_thr = profile.T_thr_rep * rng.uniform(0.5, 1.5, N_STRAINS)
        u_max = float(rng.uniform(10.0, 30.0))

        ref = simulate_case_metrics_fast(
            B0, k_arr, gamma_arr, rho_arr, mu_arr, u_max, T_thr, backend="python"
        )
        fast = simulate_case_metrics_fast(
            B0, k_arr, gamma_arr, rho_arr, mu_arr, u_max, T_thr, backend="numba"
        )
        for idx, name in enumerate(METRIC_NAMES):
            if not np.isclose(ref[idx], fast[idx], rtol=rtol, atol=atol):
                ok = False
                warnings.append(
                    f"{name}: python={ref[idx]:.6g} numba={fast[idx]:.6g} "
                    f"(rtol={rtol}, atol={atol})"
                )
    return ok, warnings


def effective_backend(requested: str, validated: bool) -> str:
    if requested == "python":
        return "python"
    if requested == "numba" and validated and NUMBA_SIMULATION_ENABLED:
        return "numba"
    return "python"
