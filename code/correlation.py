"""Paired OD600–CFU linear calibrations used for manuscript CFU-scale provenance.

E. coli uses the same eight time-aligned OD/CFU pairs as corrEcoli.py (no
interpolation; no secondary unmatched CFU series). L. lactis uses the paired
rows in corrLlactis.py.

This script asserts agreement with:
  1) multi_pathogen_simulator.py OD–CFU constants
  2) microbio_dataset.build_ode_parameter_sources_df() OD_to_CFU rows
  3) the regression fit from these paired measurements
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import linregress

from microbio_dataset import build_ode_parameter_sources_df
from multi_pathogen_simulator import (
    ECOLI_OD_TO_CFU_INTERCEPT,
    ECOLI_OD_TO_CFU_R2,
    ECOLI_OD_TO_CFU_SLOPE,
    LLACTIS_OD_TO_CFU_INTERCEPT,
    LLACTIS_OD_TO_CFU_R2,
    LLACTIS_OD_TO_CFU_SLOPE,
    experimental_calibrated_provenance,
)

# Same eight paired OD/CFU measurements as corrEcoli.py (time-aligned; no interp).
ECOLI_OD600 = np.array(
    [0.002, 0.022, 0.07, 0.4, 0.518, 0.646, 0.784, 1.36],
    dtype=float,
)
ECOLI_CFU_PER_ML = np.array(
    [
        1103333.333,
        2044166.667,
        6050000.0,
        29000000.0,
        44400000.0,
        64000000.0,
        125000000.0,
        285000000.0,
    ],
    dtype=float,
)
ECOLI_TIME_H = np.array(
    [0.0, 1.0, 2.166666667, 3.75, 4.25, 4.75, 5.25, 7.25],
    dtype=float,
)

# Same paired (time, CFU, OD600) rows as corrLlactis.py.
LLACTIS_PAIRED = np.array(
    [
        (0.0, 4154000.0, 0.034),
        (10.0, 31500000.0, 0.082),
        (13.0, 64400000.0, 0.182),
        (16.0, 105500000.0, 0.378),
        (19.0, 83933333.33, 0.664),
        (22.0, 163000000.0, 0.836),
        (25.0, 288000000.0, 1.054),
        (28.33333333, 300250000.0, 1.086),
        (30.5, 287000000.0, 1.112),
        (37.0, 434600000.0, 1.128),
        (40.0, 329166666.7, 1.108),
        (43.0, 269083333.3, 1.136),
        (46.0, 364250000.0, 1.132),
        (49.0, 288000000.0, 1.084),
        (73.25, 349400000.0, 1.082),
    ],
    dtype=float,
)

R2_ATOL = 1e-15
FIT_ATOL = 1e-9


def _fit_od_cfu(od600: np.ndarray, cfu: np.ndarray) -> tuple[float, float, float]:
    slope, intercept, r_value, _p_value, _std_err = linregress(od600, cfu)
    return float(slope), float(intercept), float(r_value**2)


def fit_ecoli_od_to_cfu() -> tuple[float, float, float]:
    assert ECOLI_OD600.shape == ECOLI_CFU_PER_ML.shape == ECOLI_TIME_H.shape
    assert ECOLI_OD600.size == 8
    return _fit_od_cfu(ECOLI_OD600, ECOLI_CFU_PER_ML)


def fit_llactis_od_to_cfu() -> tuple[float, float, float]:
    od600 = LLACTIS_PAIRED[:, 2]
    cfu = LLACTIS_PAIRED[:, 1]
    return _fit_od_cfu(od600, cfu)


def _assert_close(name: str, got: float, expected: float, atol: float) -> None:
    if not np.isclose(got, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"{name}: got={got!r}, expected={expected!r}, atol={atol}")


def assert_corr_scripts_share_paired_inputs() -> None:
    """Fail if corrEcoli.py / corrLlactis.py numerical inputs diverge from this file."""
    import corrEcoli
    import corrLlactis

    ecoli_od = np.asarray(corrEcoli.od_values, dtype=float)
    ecoli_cfu = np.asarray([cfu for _t, cfu in corrEcoli.cfu_data_1], dtype=float)
    ecoli_time = np.asarray(corrEcoli.od_time, dtype=float)
    if not np.allclose(ecoli_od, ECOLI_OD600, rtol=0.0, atol=0.0):
        raise AssertionError("corrEcoli.py od_values differ from correlation.py paired OD600")
    if not np.allclose(ecoli_cfu, ECOLI_CFU_PER_ML, rtol=0.0, atol=0.0):
        raise AssertionError("corrEcoli.py cfu_data_1 differs from correlation.py paired CFU")
    if not np.allclose(ecoli_time, ECOLI_TIME_H, rtol=0.0, atol=0.0):
        raise AssertionError("corrEcoli.py od_time differs from correlation.py paired times")

    ll = np.asarray(corrLlactis.data, dtype=float)
    if not np.allclose(ll, LLACTIS_PAIRED, rtol=0.0, atol=0.0):
        raise AssertionError("corrLlactis.py data differs from correlation.py paired rows")


def assert_three_provenance_locations_agree(
    ecoli_fit: tuple[float, float, float],
    llactis_fit: tuple[float, float, float],
) -> None:
    e_slope, e_intercept, e_r2 = ecoli_fit
    l_slope, l_intercept, l_r2 = llactis_fit

    # 1) Paired-measurement fit vs manuscript target R2 / simulator constants
    _assert_close("E. coli R2 (paired fit)", e_r2, 0.9016120870162722, R2_ATOL)
    _assert_close("E. coli slope vs simulator", e_slope, ECOLI_OD_TO_CFU_SLOPE, FIT_ATOL)
    _assert_close("E. coli intercept vs simulator", e_intercept, ECOLI_OD_TO_CFU_INTERCEPT, FIT_ATOL)
    _assert_close("E. coli R2 vs simulator", e_r2, ECOLI_OD_TO_CFU_R2, R2_ATOL)

    _assert_close("L. lactis slope vs simulator", l_slope, LLACTIS_OD_TO_CFU_SLOPE, FIT_ATOL)
    _assert_close(
        "L. lactis intercept vs simulator", l_intercept, LLACTIS_OD_TO_CFU_INTERCEPT, FIT_ATOL
    )
    _assert_close("L. lactis R2 vs simulator", l_r2, LLACTIS_OD_TO_CFU_R2, R2_ATOL)

    # 2) experimental_calibrated_provenance formulas/R2 generated from the same constants
    cal = experimental_calibrated_provenance()["od_to_cfu_calibration"]
    expected_ecoli_formula = (
        f"CFU = {ECOLI_OD_TO_CFU_SLOPE} * OD600 + ({ECOLI_OD_TO_CFU_INTERCEPT})"
    )
    expected_llactis_formula = (
        f"CFU = {LLACTIS_OD_TO_CFU_SLOPE} * OD600 + ({LLACTIS_OD_TO_CFU_INTERCEPT})"
    )
    if cal["E_coli"]["formula"] != expected_ecoli_formula:
        raise AssertionError("experimental_calibrated_provenance E. coli formula mismatch")
    if cal["E_coli"]["R2"] != ECOLI_OD_TO_CFU_R2:
        raise AssertionError("experimental_calibrated_provenance E. coli R2 mismatch")
    if cal["L_lactis"]["formula"] != expected_llactis_formula:
        raise AssertionError("experimental_calibrated_provenance L. lactis formula mismatch")
    if cal["L_lactis"]["R2"] != LLACTIS_OD_TO_CFU_R2:
        raise AssertionError("experimental_calibrated_provenance L. lactis R2 mismatch")

    # 3) microbio_dataset sources table
    sources = build_ode_parameter_sources_df()
    ecoli_row = sources.loc[sources["code_parameter"] == "E_coli_OD600"].iloc[0]
    llactis_row = sources.loc[sources["code_parameter"] == "L_lactis_OD600"].iloc[0]
    if str(ecoli_row["value_used"]) != expected_ecoli_formula:
        raise AssertionError("build_ode_parameter_sources_df E. coli formula mismatch")
    if float(ecoli_row["fit_R2"]) != ECOLI_OD_TO_CFU_R2:
        raise AssertionError("build_ode_parameter_sources_df E. coli R2 mismatch")
    if str(llactis_row["value_used"]) != expected_llactis_formula:
        raise AssertionError("build_ode_parameter_sources_df L. lactis formula mismatch")
    if float(llactis_row["fit_R2"]) != LLACTIS_OD_TO_CFU_R2:
        raise AssertionError("build_ode_parameter_sources_df L. lactis R2 mismatch")


def main() -> None:
    assert_corr_scripts_share_paired_inputs()
    ecoli_fit = fit_ecoli_od_to_cfu()
    llactis_fit = fit_llactis_od_to_cfu()
    assert_three_provenance_locations_agree(ecoli_fit, llactis_fit)

    print("E. coli paired OD–CFU fit (n=8, no interpolation)")
    print(f"  slope={ecoli_fit[0]}, intercept={ecoli_fit[1]}, R2={ecoli_fit[2]}")
    print("L. lactis paired OD–CFU fit")
    print(f"  slope={llactis_fit[0]}, intercept={llactis_fit[1]}, R2={llactis_fit[2]}")
    print("All three OD–CFU provenance locations agree.")
    print(f"Workspace root: {Path(__file__).resolve().parent}")


if __name__ == "__main__":
    main()
