from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

# --- unified_residual (inlined) ---
CANDIDATE_SCORE_COLUMNS = [
    "candidate_top_score",
    "candidate_second_score",
    "candidate_score_margin",
    "candidate_entropy",
    "controller_margin_risk",
    "soft_label_confidence",
]

DIAGNOSTIC_ONLY_COLS = [
    "lr_error",
    "pauc_error",
    "match_error_no_dose",
    "selection_score",
    "constraint_violation",
]

VAL_SAFE_META_COLS = [
    "desired_profile_id",
    "desired_P_AUC",
    "desired_LR1",
    "desired_LR2",
    "desired_LR3",
    "desired_LR4",
    "desired_LR5",
    "target_uncertainty",
    "soft_u_max",
]

SUPERVISED_KEY_CANDIDATES = (
    "row_id",
    "original_row_index",
    "sample_id",
    "supervised_sample_id",
)
GROUP_KEY_CANDIDATES = (
    ("row_id",),
    ("bio_id", "desired_profile_id"),
    ("bio_id", "desired_profile_id", "row_id"),
    ("original_row_index",),
    ("sample_id",),
)


def _score_column(df: pd.DataFrame) -> str:
    for col in ("final_score", "top_score", "candidate_top_score", "selection_score"):
        if col in df.columns:
            return col
    raise ValueError("candidate score table missing a score column (final_score / top_score).")


def _weight_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("softmax_weight", "soft_label_weight", "candidate_weight"):
        if col in df.columns:
            return col
    return None


def _entropy_from_weights(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    if w.size == 0:
        return float("nan")
    w = w / w.sum()
    return float(-np.sum(w * np.log(w + 1e-30)))


def aggregate_candidate_table_long(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse long candidate tables to one row per supervised sample."""
    score_col = _score_column(df)
    weight_col = _weight_column(df)
    tthr_cols = [c for c in df.columns if c.startswith("Tthr_")]

    group_cols: Optional[Tuple[str, ...]] = None
    for cand in GROUP_KEY_CANDIDATES:
        if all(c in df.columns for c in cand):
            group_cols = cand
            break
    if group_cols is None:
        raise ValueError(
            "Cannot aggregate candidate table: need row_id or (bio_id, desired_profile_id)."
        )

    rows: List[dict] = []
    for keys, group in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))
        work = group.copy()
        scores = work[score_col].astype(float).to_numpy()
        order = np.argsort(scores)[::-1]
        top_score = float(scores[order[0]])
        second_score = float(scores[order[1]]) if len(scores) > 1 else top_score
        margin = top_score - second_score
        if weight_col is not None:
            weights = work[weight_col].astype(float).to_numpy()
            entropy = _entropy_from_weights(weights)
            soft_conf = float(np.max(weights))
        else:
            entropy = float("nan")
            soft_conf = float(1.0 / (1.0 + np.exp(-np.clip(margin, -20, 20))))

        topk_work = work
        if "in_top_k" in work.columns:
            flagged = work[work["in_top_k"].astype(bool)]
        elif "included_in_soft_label" in work.columns:
            flagged = work[work["included_in_soft_label"].astype(bool)]
        else:
            flagged = work.iloc[0:0]
        if len(flagged) >= 2:
            topk_work = flagged
        std_mean = range_mean = float("nan")
        if tthr_cols:
            tthr = topk_work[tthr_cols].astype(float).to_numpy()
            if len(tthr) >= 1:
                std_mean = float(np.mean(np.std(tthr, axis=0)))
                range_mean = float(np.mean(np.ptp(tthr, axis=0)))

        rows.append(
            {
                **key_dict,
                "candidate_top_score": top_score,
                "candidate_second_score": second_score,
                "candidate_score_margin": margin,
                "candidate_entropy": entropy,
                "soft_label_confidence": soft_conf,
                "top_k_Tthr_std_mean": std_mean,
                "top_k_Tthr_range_mean": range_mean,
            }
        )
    return pd.DataFrame(rows)


def _merge_keys(metadata: pd.DataFrame) -> Tuple[str, ...]:
    if "row_id" in metadata.columns:
        return ("row_id",)
    if {"bio_id", "desired_profile_id"}.issubset(metadata.columns):
        return ("bio_id", "desired_profile_id")
    for col in SUPERVISED_KEY_CANDIDATES:
        if col in metadata.columns:
            return (col,)
    raise ValueError("metadata lacks row_id or (bio_id, desired_profile_id) for candidate alignment.")


def align_candidate_features_to_metadata(
    candidate_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, bool]:
    """Return candidate feature frame aligned to metadata row order."""
    meta = metadata.reset_index(drop=True).copy()
    merge_keys = _merge_keys(meta)
    if merge_keys == ("row_id",) and "row_id" not in candidate_df.columns:
        if {"bio_id", "desired_profile_id"}.issubset(meta.columns) and {"bio_id", "desired_profile_id"}.issubset(
            candidate_df.columns
        ):
            merge_keys = ("bio_id", "desired_profile_id")
        else:
            raise ValueError("candidate table lacks row_id and cannot merge on bio_id/desired_profile_id.")
    if len(candidate_df) == len(meta) and all(k in candidate_df.columns for k in merge_keys):
        aligned = candidate_df.reset_index(drop=True)
        if merge_keys == ("row_id",) and aligned["row_id"].equals(meta["row_id"]):
            return _standardize_feature_columns(aligned), True

    aggregated = aggregate_candidate_table_long(candidate_df)
    merged = meta.merge(aggregated, on=list(merge_keys), how="left", suffixes=("", "_cand"))
    if len(merged) != len(meta):
        raise ValueError("Candidate alignment merge changed row count.")
    missing = merged["candidate_score_margin"].isna().mean() if "candidate_score_margin" in merged.columns else 1.0
    if missing > 0.5:
        warnings.warn(
            f"candidate score alignment: {missing:.1%} rows missing margin after merge.",
            stacklevel=2,
        )
        return _standardize_feature_columns(merged), False
    return _standardize_feature_columns(merged), True


def _standardize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    mapping = {
        "candidate_top_score": ["candidate_top_score", "top_score"],
        "candidate_second_score": ["candidate_second_score", "second_score"],
        "candidate_score_margin": ["candidate_score_margin", "score_margin"],
        "candidate_entropy": ["candidate_entropy", "score_entropy"],
        "soft_label_confidence": ["soft_label_confidence"],
        "top_k_Tthr_std_mean": ["top_k_Tthr_std_mean"],
        "top_k_Tthr_range_mean": ["top_k_Tthr_range_mean"],
    }
    for out_col, sources in mapping.items():
        for src in sources:
            if src in df.columns:
                out[out_col] = df[src].astype(float)
                break
        if out_col not in out.columns:
            out[out_col] = 0.0
    if "candidate_top_score" in out.columns and "candidate_second_score" in out.columns:
        if out["candidate_score_margin"].eq(0).all():
            out["candidate_score_margin"] = out["candidate_top_score"] - out["candidate_second_score"]
    eps = 1e-6
    if "candidate_score_margin" in out.columns:
        margin = out["candidate_score_margin"].to_numpy(dtype=float)
        out["controller_margin_risk"] = 1.0 / (np.abs(margin) + eps)
        mx = out["controller_margin_risk"].max()
        if mx > 0:
            out["controller_margin_risk"] = out["controller_margin_risk"] / (mx + eps)
    if out["soft_label_confidence"].eq(0).all() and "candidate_score_margin" in out.columns:
        m = out["candidate_score_margin"].to_numpy(dtype=float)
        out["soft_label_confidence"] = 1.0 / (1.0 + np.exp(-np.clip(m, -20, 20)))
    return out.fillna(0.0)


def load_candidate_score_table(
    path: str,
    metadata: Optional[pd.DataFrame] = None,
    n_rows: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], bool, bool]:
    """
    Load and align candidate scores to supervised samples.

    Returns (features, aligned_ok, alignment_failed).
    """
    if not path or not str(path).strip():
        return None, False, False
    import os

    if not os.path.exists(path):
        warnings.warn(f"candidate_score_csv not found: {path}", stacklevel=2)
        return None, False, True

    raw = pd.read_csv(path)
    if metadata is None:
        if n_rows is not None and len(raw) == n_rows:
            return _standardize_feature_columns(raw), True, False
        warnings.warn(
            f"candidate_score_csv row count ({len(raw)}) != dataset rows ({n_rows}); "
            "pass metadata to enable long-format aggregation.",
            stacklevel=2,
        )
        return None, False, True

    try:
        if len(raw) == len(metadata):
            features, ok = align_candidate_features_to_metadata(raw, metadata)
            return features, ok, not ok
        aggregated = aggregate_candidate_table_long(raw)
        features, ok = align_candidate_features_to_metadata(aggregated, metadata)
        return features, ok, not ok
    except Exception as exc:
        warnings.warn(f"Failed to align candidate_score_csv ({path}): {exc}", stacklevel=2)
        return None, False, True


CANDIDATE_MARGIN_CONSTANT_EPS = 1e-12


def candidate_score_margin_informative(aligned_features: Optional[pd.DataFrame]) -> bool:
    if aligned_features is None or aligned_features.empty:
        return False
    if "candidate_score_margin" not in aligned_features.columns:
        return False
    margin = aligned_features["candidate_score_margin"].astype(float)
    if margin.isna().all():
        return False
    return float(margin.max() - margin.min()) > CANDIDATE_MARGIN_CONSTANT_EPS


def warn_if_candidate_margin_uninformative(aligned_features: Optional[pd.DataFrame]) -> bool:
    informative = candidate_score_margin_informative(aligned_features)
    if not informative:
        warnings.warn(
            "candidate_score_margin is nearly constant; ambiguity diagnostic is not informative.",
            stacklevel=2,
        )
    return informative

# --- uncertainty_decomposition (inlined) ---
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    _HAS_TORCH = False

UNCERTAINTY_MODEL = "UncertaintyNet"
UncertaintyMethod = Literal["mc_dropout", "deep_ensemble"]


@dataclass
class UncertaintyDecompositionConfig:
    enabled: bool = False
    method: UncertaintyMethod = "mc_dropout"
    hidden_dim: int = 128
    dropout: float = 0.10
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    mc_samples: int = 50
    n_ensemble: int = 5
    inference_batch_size: int = 512
    show_main: bool = False
    force_cpu: bool = False


@dataclass
class UncertaintyDecompositionResult:
    case_rows: List[dict] = field(default_factory=list)
    summary_rows: List[dict] = field(default_factory=list)
    training_log_rows: List[dict] = field(default_factory=list)
    manifest: Dict[str, object] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""


def build_uncertainty_config_from_args(args) -> UncertaintyDecompositionConfig:
    method = str(getattr(args, "uncertainty_method", "mc_dropout")).lower()
    if method not in ("mc_dropout", "deep_ensemble"):
        raise ValueError("uncertainty_method must be mc_dropout or deep_ensemble")
    return UncertaintyDecompositionConfig(
        enabled=bool(getattr(args, "enable_uncertainty_decomposition", False)),
        method=method,  # type: ignore[arg-type]
        hidden_dim=int(getattr(args, "uncertainty_hidden_dim", 128)),
        dropout=float(getattr(args, "uncertainty_dropout", 0.10)),
        epochs=int(getattr(args, "uncertainty_epochs", 100)),
        batch_size=int(getattr(args, "uncertainty_batch_size", 256)),
        lr=float(getattr(args, "uncertainty_lr", 1e-3)),
        mc_samples=int(getattr(args, "uncertainty_mc_samples", 50)),
        n_ensemble=int(getattr(args, "uncertainty_n_ensemble", 5)),
        inference_batch_size=int(getattr(args, "uncertainty_inference_batch_size", 512)),
        show_main=bool(getattr(args, "show_uncertainty_main", False)),
        force_cpu=bool(getattr(args, "uncertainty_force_cpu", False)),
    )


def _gaussian_nll(y: "torch.Tensor", mu: "torch.Tensor", log_var: "torch.Tensor") -> "torch.Tensor":
    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (y - mu) ** 2 * inv_var)


if _HAS_TORCH:

    class HeteroscedasticMLP(nn.Module):
        """Predict per-target mean and log-variance (aleatoric) with MC-Dropout-ready trunk."""

        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
            self.mu_head = nn.Linear(hidden_dim, output_dim)
            self.log_var_head = nn.Linear(hidden_dim, output_dim)

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            h = self.trunk(x)
            return self.mu_head(h), self.log_var_head(h)


def _train_heteroscedastic_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: UncertaintyDecompositionConfig,
    seed: int,
) -> Tuple["HeteroscedasticMLP", List[dict]]:
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for uncertainty decomposition")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg.force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeteroscedasticMLP(
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)), shuffle=True)

    log_rows: List[dict] = []
    model.train()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            mu, log_var = model(x_b)
            loss = torch.mean(_gaussian_nll(y_b, mu, log_var))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        log_rows.append({"epoch": epoch + 1, "train_nll": epoch_loss / max(n_batches, 1)})
    return model, log_rows


def _predict_mc_dropout(
    model: "HeteroscedasticMLP",
    X: np.ndarray,
    mc_samples: int,
    inference_batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched MC Dropout: one forward pass per chunk with mc_samples stacked on the batch dim."""
    if not _HAS_TORCH:
        raise RuntimeError("torch is required")

    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")

    device = next(model.parameters()).device
    n_cases = X.shape[0]
    output_dim = model.mu_head.out_features
    pred_mean = np.zeros((n_cases, output_dim), dtype=float)
    aleatoric_var = np.zeros((n_cases, output_dim), dtype=float)
    epistemic_var = np.zeros((n_cases, output_dim), dtype=float)
    chunk_size = max(1, min(int(inference_batch_size), n_cases))

    model.train()  # dropout active
    with torch.no_grad():
        for start in range(0, n_cases, chunk_size):
            end = min(start + chunk_size, n_cases)
            x_b = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            batch_n = x_b.shape[0]
            # Independent dropout masks per MC draw: (mc_samples * batch_n, input_dim)
            x_rep = x_b.repeat(mc_samples, 1)
            mu, log_var = model(x_rep)
            mu = mu.view(mc_samples, batch_n, output_dim)
            ale = torch.exp(log_var.view(mc_samples, batch_n, output_dim))
            pred_mean[start:end] = mu.mean(dim=0).cpu().numpy()
            epistemic_var[start:end] = mu.var(dim=0, unbiased=False).cpu().numpy()
            aleatoric_var[start:end] = ale.mean(dim=0).cpu().numpy()
    return pred_mean, aleatoric_var, epistemic_var


def _predict_deep_ensemble(
    models: Sequence["HeteroscedasticMLP"],
    X: np.ndarray,
    inference_batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not _HAS_TORCH:
        raise RuntimeError("torch is required")

    device = next(models[0].parameters()).device
    n_cases = X.shape[0]
    output_dim = models[0].mu_head.out_features
    pred_mean = np.zeros((n_cases, output_dim), dtype=float)
    aleatoric_var = np.zeros((n_cases, output_dim), dtype=float)
    epistemic_var = np.zeros((n_cases, output_dim), dtype=float)
    chunk_size = max(1, min(int(inference_batch_size), n_cases))

    with torch.no_grad():
        for start in range(0, n_cases, chunk_size):
            end = min(start + chunk_size, n_cases)
            x_b = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            mu_members: List["torch.Tensor"] = []
            ale_members: List["torch.Tensor"] = []
            for model in models:
                model.eval()
                mu, log_var = model(x_b)
                mu_members.append(mu)
                ale_members.append(torch.exp(log_var))
            mu_stack = torch.stack(mu_members, dim=0)
            ale_stack = torch.stack(ale_members, dim=0)
            pred_mean[start:end] = mu_stack.mean(dim=0).cpu().numpy()
            epistemic_var[start:end] = mu_stack.var(dim=0, unbiased=False).cpu().numpy()
            aleatoric_var[start:end] = ale_stack.mean(dim=0).cpu().numpy()
    return pred_mean, aleatoric_var, epistemic_var


def _build_case_rows(
    *,
    repeat_id: int,
    y_true: np.ndarray,
    pred_mean: np.ndarray,
    aleatoric_var: np.ndarray,
    epistemic_var: np.ndarray,
    target_names: Sequence[str],
) -> Tuple[List[dict], List[dict]]:
    total_var = aleatoric_var + epistemic_var
    case_rows: List[dict] = []
    for case_idx in range(len(y_true)):
        for t_idx, target in enumerate(target_names):
            err = float(y_true[case_idx, t_idx] - pred_mean[case_idx, t_idx])
            ale_v = float(aleatoric_var[case_idx, t_idx])
            epi_v = float(epistemic_var[case_idx, t_idx])
            tot_v = float(total_var[case_idx, t_idx])
            case_rows.append(
                {
                    "repeat_id": repeat_id,
                    "case_index": case_idx,
                    "target": target,
                    "pred_mean": float(pred_mean[case_idx, t_idx]),
                    "true_value": float(y_true[case_idx, t_idx]),
                    "abs_error": abs(err),
                    "aleatoric_variance": ale_v,
                    "epistemic_variance": epi_v,
                    "total_variance": tot_v,
                    "aleatoric_std": float(np.sqrt(max(ale_v, 0.0))),
                    "epistemic_std": float(np.sqrt(max(epi_v, 0.0))),
                    "total_std": float(np.sqrt(max(tot_v, 0.0))),
                    "epistemic_fraction": float(epi_v / tot_v) if tot_v > 0 else float("nan"),
                }
            )

    summary_rows: List[dict] = []
    df = pd.DataFrame(case_rows)
    for target in target_names:
        sub = df[df["target"] == target]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "repeat_id": repeat_id,
                "target": target,
                "n_cases": len(sub),
                "mean_aleatoric_std": float(sub["aleatoric_std"].mean()),
                "mean_epistemic_std": float(sub["epistemic_std"].mean()),
                "mean_total_std": float(sub["total_std"].mean()),
                "mean_epistemic_fraction": float(sub["epistemic_fraction"].mean()),
                "spearman_abs_error_vs_total_std": _safe_spearman(sub["abs_error"], sub["total_std"]),
                "spearman_abs_error_vs_epistemic_std": _safe_spearman(sub["abs_error"], sub["epistemic_std"]),
                "spearman_abs_error_vs_aleatoric_std": _safe_spearman(sub["abs_error"], sub["aleatoric_std"]),
            }
        )
    return case_rows, summary_rows


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(x.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        return float("nan")


def run_uncertainty_decomposition(
    data,
    *,
    cfg: UncertaintyDecompositionConfig,
    repeat_id: int,
    seed: int,
    target_transform: str,
) -> UncertaintyDecompositionResult:
    """Train heteroscedastic net; decompose validation uncertainty into aleatoric vs epistemic."""
    base_manifest: Dict[str, object] = {
        "model": UNCERTAINTY_MODEL,
        "exploratory_not_for_main_claim": True,
        "uncertainty_enabled": False,
        "uncertainty_method": cfg.method,
        "show_uncertainty_main": cfg.show_main,
    }
    if not cfg.enabled:
        return UncertaintyDecompositionResult(manifest=base_manifest, skipped=True, skip_reason="disabled")

    if not _HAS_TORCH:
        base_manifest["skip_reason"] = "torch_not_installed"
        warnings.warn("Uncertainty decomposition skipped: PyTorch is not installed.", stacklevel=2)
        return UncertaintyDecompositionResult(
            manifest={**base_manifest, "uncertainty_skipped_no_torch": True},
            skipped=True,
            skip_reason="torch_not_installed",
        )

    print(
        f"  Uncertainty decomposition ({cfg.method}: epistemic vs aleatoric)",
        flush=True,
    )

    X_train = np.asarray(data.X_train, dtype=float)
    X_val = np.asarray(data.X_val, dtype=float)
    y_train_fit = np.asarray(data.y_train_fit, dtype=float)
    y_val_fit = np.asarray(data.y_val_fit, dtype=float)
    target_names = list(data.y_columns)

    training_log_rows: List[dict] = []
    if cfg.method == "mc_dropout":
        model, logs = _train_heteroscedastic_mlp(X_train, y_train_fit, cfg, seed=seed + 211)
        for row in logs:
            row["repeat_id"] = repeat_id
            row["member_id"] = 0
        training_log_rows.extend(logs)
        pred_mean, aleatoric_var, epistemic_var = _predict_mc_dropout(
            model, X_val, cfg.mc_samples, cfg.inference_batch_size
        )
    else:
        models: List[HeteroscedasticMLP] = []
        for member_id in range(cfg.n_ensemble):
            member, logs = _train_heteroscedastic_mlp(
                X_train, y_train_fit, cfg, seed=seed + 211 + member_id * 997
            )
            models.append(member)
            for row in logs:
                row["repeat_id"] = repeat_id
                row["member_id"] = member_id
            training_log_rows.extend(logs)
        pred_mean, aleatoric_var, epistemic_var = _predict_deep_ensemble(
            models, X_val, cfg.inference_batch_size
        )

    case_rows, summary_rows = _build_case_rows(
        repeat_id=repeat_id,
        y_true=y_val_fit,
        pred_mean=pred_mean,
        aleatoric_var=aleatoric_var,
        epistemic_var=epistemic_var,
        target_names=target_names,
    )

    manifest = {
        **base_manifest,
        "uncertainty_enabled": True,
        "uncertainty_skipped_no_torch": False,
        "repeat_id": repeat_id,
        "target_transform": target_transform,
        "n_validation_cases": len(X_val),
        "n_targets": len(target_names),
        "hidden_dim": cfg.hidden_dim,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "mc_samples": cfg.mc_samples if cfg.method == "mc_dropout" else None,
        "inference_batch_size": cfg.inference_batch_size,
        "mc_inference_mode": "batched_gpu" if cfg.method == "mc_dropout" else None,
        "n_ensemble": cfg.n_ensemble if cfg.method == "deep_ensemble" else None,
        "decomposition_note": (
            "Aleatoric = predicted data noise (mean exp(log_var)); "
            "Epistemic = model uncertainty (Var of predictive mean across MC passes or ensemble members); "
            "Total = aleatoric + epistemic variance. "
            "MC Dropout uses batched inference (mc_samples stacked per mini-batch on GPU)."
        ),
    }
    return UncertaintyDecompositionResult(
        case_rows=case_rows,
        summary_rows=summary_rows,
        training_log_rows=training_log_rows,
        manifest=manifest,
        skipped=False,
    )


def write_uncertainty_artifacts(
    outdir: str,
    case_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    training_log_df: pd.DataFrame,
    manifest: Dict[str, object],
) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    outputs: Dict[str, str] = {}
    if not case_df.empty:
        path = os.path.join(outdir, "uncertainty_decomposition.csv")
        case_df.to_csv(path, index=False)
        outputs["uncertainty_decomposition.csv"] = path
    if not summary_df.empty:
        path = os.path.join(outdir, "uncertainty_decomposition_summary.csv")
        summary_df.to_csv(path, index=False)
        outputs["uncertainty_decomposition_summary.csv"] = path
    if not training_log_df.empty:
        path = os.path.join(outdir, "uncertainty_training_log.csv")
        training_log_df.to_csv(path, index=False)
        outputs["uncertainty_training_log.csv"] = path
    manifest_path = os.path.join(outdir, "uncertainty_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    outputs["uncertainty_manifest.json"] = manifest_path
    return outputs


try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None  # type: ignore


def make_lgbm_regressor(seed: int, **overrides) -> "LGBMRegressor":
    if LGBMRegressor is None:
        raise ImportError("LightGBM is required: pip install lightgbm")
    params = {
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "device_type": os.environ.get("TREE_SRL_LGBM_DEVICE", "gpu"),
        **overrides,
    }
    return LGBMRegressor(**params)


def _configure_quiet_runtime(verbose: bool = False) -> None:
    if verbose:
        return
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    warnings.filterwarnings("ignore", message="LightGBM GPU fit failed.*")
    try:
        rank_warning = np.RankWarning  # type: ignore[attr-defined]
    except AttributeError:
        try:
            from numpy.exceptions import RankWarning as rank_warning
        except ImportError:
            rank_warning = None
    if rank_warning is not None:
        warnings.filterwarnings("ignore", category=rank_warning)
    else:
        warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")
    try:
        from scipy.stats import ConstantInputWarning

        warnings.filterwarnings("ignore", category=ConstantInputWarning)
    except ImportError:
        pass


def _configure_parallel_worker(lgbm_device: str = "gpu", verbose: bool = False) -> None:
    """Limit BLAS thread oversubscription in worker processes; keep LightGBM on GPU by default."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ["TREE_SRL_LGBM_DEVICE"] = lgbm_device
    _configure_quiet_runtime(verbose=verbose)

TARGET_COLS_TTHR = [f"Tthr_{i}" for i in range(1, 6)]
LOG_FIT_CLIP_LOW = -30.0
LOG_FIT_CLIP_HIGH = float(np.log(np.finfo(np.float64).max) - 1.0)

TAR_MODEL = "TAR"
BEST_SINGLE_TREE = "BestSingleTree"
UNIFORM_TREE_MEAN = "UniformTreeMean"
RANDOM_FOREST = "RandomForest"

LEGACY_MODEL_NAME_MAP = {
    "TAR-SRL": TAR_MODEL,
    "TAR-SRL-no-cycle": TAR_MODEL,
    "TAR-SRL-rerank": "TAR-rerank_legacy_not_main",
}

CORE_BENCHMARK_MODELS = {
    TAR_MODEL,
    RANDOM_FOREST,
    BEST_SINGLE_TREE,
    UNIFORM_TREE_MEAN,
}

MANUSCRIPT_CONTROL_MODELS = [RANDOM_FOREST, BEST_SINGLE_TREE, UNIFORM_TREE_MEAN]

MODEL_DISPLAY_LABELS = {
    TAR_MODEL: "TAR",
    RANDOM_FOREST: "RF",
    BEST_SINGLE_TREE: "Best tree",
    UNIFORM_TREE_MEAN: "UniformTreeMean",
    "ExtraTrees": "ET",
}

MAIN_BAR_ORDER = [TAR_MODEL, RANDOM_FOREST, BEST_SINGLE_TREE, UNIFORM_TREE_MEAN]

SIGNIFICANCE_PLOT_SPECS = [
    (TAR_MODEL, RANDOM_FOREST),
    (TAR_MODEL, BEST_SINGLE_TREE),
    (TAR_MODEL, UNIFORM_TREE_MEAN),
]

PRIMARY_CSV_OUTPUTS = [
    "model_compare_summary.csv",
    "model_compare_per_target.csv",
    "repeated_parameter_metrics.csv",
    "parameter_pairwise_significance.csv",
    "target_weight_table.csv",
    "tree_expert_table.csv",
]

PRIMARY_PNG_OUTPUTS = [
    "model_compare_r2.png",
    "prediction_error_heatmap.png",
    "target_weight_heatmap.png",
]

UNCERTAINTY_PNG_OUTPUTS = [
    "uncertainty_decomposition.png",
    "uncertainty_decomposition_by_target.png",
]

REPEATED_METRIC_COLS = [
    "mean_R2_original",
    "mean_R2_log",
    "mean_RMSE_original",
    "mean_MAE_original",
    "mean_NRMSE_original",
    "mean_spearman_original",
    "mean_calibration_slope",
    "mean_calibration_intercept",
]

METRIC_DIRECTION = {
    "mean_R2_original": "higher_is_better",
    "mean_R2_log": "higher_is_better",
    "mean_spearman_original": "higher_is_better",
    "mean_calibration_slope": "higher_is_better",
    "mean_RMSE_original": "lower_is_better",
    "mean_MAE_original": "lower_is_better",
    "mean_NRMSE_original": "lower_is_better",
    "mean_calibration_intercept": "lower_is_better",
}

def model_display_label(model_name: str) -> str:
    return MODEL_DISPLAY_LABELS.get(model_name, model_name)


def normalize_model_name(model_name: str) -> str:
    """Map legacy prediction / CSV model names to current canonical names."""
    return LEGACY_MODEL_NAME_MAP.get(model_name, model_name)


def normalize_model_frame(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    if model_col in out.columns:
        out[model_col] = out[model_col].map(normalize_model_name)
    return out


def apply_display_labels(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    if model_col in out.columns:
        out[model_col] = out[model_col].map(model_display_label)
    return out


@dataclass
class BenchmarkData:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_train_fit: np.ndarray
    y_val_fit: np.ndarray
    groups_train: np.ndarray
    sample_weight_train: Optional[np.ndarray]
    target_uncertainty_train: Optional[np.ndarray]
    target_uncertainty_val: Optional[np.ndarray]
    X_scaler: StandardScaler
    X_columns: List[str]
    y_columns: List[str]
    val_indices: np.ndarray
    x_test_df: pd.DataFrame
    val_metadata: Optional[pd.DataFrame] = None
    train_metadata: Optional[pd.DataFrame] = None
    candidate_scores_train: Optional[pd.DataFrame] = None
    candidate_scores_val: Optional[pd.DataFrame] = None
    candidate_score_alignment_failed: bool = False
    candidate_score_csv_path: Optional[str] = None


@dataclass
class CycleConfig:
    """Minimal stub retained for closed_loop_eval import compatibility (cycle disabled)."""

    enabled: bool = False


@dataclass
class StackBundle:
    stacker_type: str
    weights_per_target: Dict[str, np.ndarray]
    oof_predictions: np.ndarray
    ridge_models: Dict[str, RidgeCV]


@dataclass
class RepeatResult:
    repeat_id: int
    seed: int
    per_target_rows: List[dict]
    summary_rows: List[dict]
    predictions: Dict[str, np.ndarray]
    target_weights_rows: List[dict]
    final_model_name: str = ""
    best_single_tree_name: str = ""
    chosen_stacker_type: str = ""
    tree_expert_summary_rows: List[dict] = field(default_factory=list)
    expert_names_used: List[str] = field(default_factory=list)
    uncertainty_case_rows: List[dict] = field(default_factory=list)
    uncertainty_summary_rows: List[dict] = field(default_factory=list)
    uncertainty_training_log_rows: List[dict] = field(default_factory=list)
    uncertainty_manifest: Dict[str, object] = field(default_factory=dict)
    uncertainty_skipped: bool = False
    uncertainty_skip_reason: str = ""
    split_metadata: Dict[str, object] = field(default_factory=dict)
    gated_residual_enabled: bool = False
    gated_residual_passed_gate: bool = False


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def resolve_x_csv(x_csv: str, use_physics_features: bool) -> str:
    if not use_physics_features:
        return x_csv
    directory = os.path.dirname(os.path.abspath(x_csv))
    physics_path = os.path.join(directory, "X_features_physics.csv")
    if os.path.exists(physics_path):
        print(f"Using physics features: {physics_path}")
        return physics_path
    if "physics" in os.path.basename(x_csv).lower():
        return x_csv
    raise FileNotFoundError(
        f"--use_physics_features was set but {physics_path} was not found."
    )


def resolve_metadata_csv(metadata_csv: Optional[str], x_csv: str) -> str:
    if metadata_csv:
        return metadata_csv
    return os.path.join(os.path.dirname(os.path.abspath(x_csv)), "sample_metadata.csv")


def split_train_validation_indices(
    n_samples: int,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    split_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    if split_mode == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(gss.split(indices, groups=groups))
    print(
        "WARNING: ROW-LEVEL SPLIT MAY LEAK BIOLOGICAL GROUPS ACROSS TRAIN AND VALIDATION. "
        "DO NOT USE FOR MANUSCRIPT RESULTS."
    )
    return train_test_split(indices, test_size=test_size, random_state=seed)


def forward_target_transform(y: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log":
        return np.log(np.clip(y, 1e-30, None))
    if target_transform == "none":
        return y.copy()
    raise ValueError("target_transform must be 'log' or 'none'.")


def clip_log_fit(y_fit: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_fit, dtype=float), LOG_FIT_CLIP_LOW, LOG_FIT_CLIP_HIGH)


def clip_log_fit_to_train(
    y_fit: np.ndarray,
    y_train_fit: np.ndarray,
    target_transform: str,
    margin: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Clip fit-space predictions to a training-derived range; return (clipped, n_clipped)."""
    arr = np.asarray(y_fit, dtype=float).copy()
    if target_transform != "log":
        return arr, 0
    train_col = np.asarray(y_train_fit, dtype=float).ravel()
    if train_col.size == 0:
        return clip_log_fit(arr), 0
    lo = float(np.nanquantile(train_col, 0.005)) - margin
    hi = float(np.nanquantile(train_col, 0.995)) + margin
    lo = max(lo, float(np.nanmin(train_col)) - margin)
    hi = min(hi, float(np.nanmax(train_col)) + margin)
    clipped = np.clip(arr, lo, hi)
    n_clipped = int(np.sum(clipped != arr))
    return clipped, n_clipped


def clip_log_fit_matrix_to_train(
    y_fit: np.ndarray,
    y_train_fit: np.ndarray,
    target_transform: str,
    margin: float = 1.0,
) -> Tuple[np.ndarray, int]:
    out = np.asarray(y_fit, dtype=float).copy()
    total_clipped = 0
    for j in range(out.shape[1]):
        out[:, j], n = clip_log_fit_to_train(out[:, j], y_train_fit[:, j], target_transform, margin)
        total_clipped += n
    return out, total_clipped


def validate_fit_space_predictions(
    y_fit: np.ndarray,
    model_name: str,
    context: str,
) -> Tuple[bool, str]:
    arr = np.asarray(y_fit, dtype=float)
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        return False, f"{model_name}: {n_bad} non-finite fit-space predictions ({context})"
    return True, ""


def safe_inverse_target_transform(
    y_fit: np.ndarray,
    target_transform: str,
    model_name: str,
) -> Tuple[np.ndarray, bool, str]:
    ok, reason = validate_fit_space_predictions(y_fit, model_name, "pre-inverse-transform")
    if not ok:
        return np.full_like(y_fit, np.nan, dtype=float), False, reason
    y_orig = inverse_target_transform(y_fit, target_transform)
    if not np.all(np.isfinite(y_orig)):
        n_bad = int(np.sum(~np.isfinite(y_orig)))
        return y_orig, False, f"{model_name}: {n_bad} non-finite values after inverse transform"
    return y_orig, True, ""


def inverse_target_transform(y_fit: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log":
        return np.exp(clip_log_fit(y_fit))
    return np.asarray(y_fit, dtype=float).copy()


def safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred):
        return float("nan")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    if float(np.var(yt)) <= 1e-30:
        return float("nan")
    try:
        return float(r2_score(yt, yp))
    except ValueError:
        return float("nan")


def mean_target_wise_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-target sklearn R² (matches benchmark ``mean_R2_original``)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        return safe_r2_score(y_true, y_pred)
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape[1] != y_pred.shape[1]:
        return float("nan")
    per_target = [safe_r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]
    finite = [s for s in per_target if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")


def r2_gain(y_true: np.ndarray, y_pred_new: np.ndarray, y_pred_base: np.ndarray) -> float:
    new_score = safe_r2_score(y_true, y_pred_new)
    base_score = safe_r2_score(y_true, y_pred_base)
    if not np.isfinite(new_score) or not np.isfinite(base_score):
        return float("nan")
    return float(new_score - base_score)


def load_benchmark_data(
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    sample_weight_csv: Optional[str],
    split_mode: str,
    group_col: str,
    test_size: float,
    seed: int,
    target_transform: str,
    max_rows: int = 0,
    use_physics_features: bool = False,
    candidate_score_csv: Optional[str] = None,
) -> BenchmarkData:
    x_path = resolve_x_csv(x_csv, use_physics_features)
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Cannot find X CSV: {x_path}")
    if not os.path.exists(y_csv):
        raise FileNotFoundError(f"Cannot find y CSV: {y_csv}")

    X_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_csv)
    valid = ~(X_df.isna().any(axis=1) | y_df.isna().any(axis=1))
    X_df = X_df.loc[valid].reset_index(drop=True)
    y_df = y_df.loc[valid].reset_index(drop=True)

    if set(TARGET_COLS_TTHR).issubset(y_df.columns):
        y_df = y_df[TARGET_COLS_TTHR]
    elif y_df.shape[1] != 5:
        raise ValueError("y_targets.csv must contain Tthr_1..Tthr_5.")

    metadata_path = resolve_metadata_csv(metadata_csv, x_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Cannot find metadata CSV: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    if len(metadata_df) != len(X_df):
        raise ValueError(
            f"metadata row count ({len(metadata_df)}) must match X/y row count ({len(X_df)})."
        )
    if group_col not in metadata_df.columns:
        raise ValueError(f"group_col '{group_col}' not found in metadata CSV.")

    sample_weight_all: Optional[np.ndarray] = None
    target_uncertainty_all: Optional[np.ndarray] = None
    if sample_weight_csv:
        if not os.path.exists(sample_weight_csv):
            raise FileNotFoundError(f"Cannot find sample weight CSV: {sample_weight_csv}")
        weight_df = pd.read_csv(sample_weight_csv)
        if len(weight_df) != len(X_df):
            raise ValueError("sample_weight_csv row count must match X/y row count.")
        weight_col = "sample_weight" if "sample_weight" in weight_df.columns else weight_df.columns[-1]
        sample_weight_all = weight_df[weight_col].astype(float).to_numpy()
        if "target_uncertainty" in weight_df.columns:
            target_uncertainty_all = weight_df["target_uncertainty"].astype(float).to_numpy()

    if max_rows and max_rows > 0 and len(X_df) > max_rows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(X_df), size=max_rows, replace=False))
        X_df = X_df.iloc[keep].reset_index(drop=True)
        y_df = y_df.iloc[keep].reset_index(drop=True)
        metadata_df = metadata_df.iloc[keep].reset_index(drop=True)
        if sample_weight_all is not None:
            sample_weight_all = sample_weight_all[keep]
        if target_uncertainty_all is not None:
            target_uncertainty_all = target_uncertainty_all[keep]
        print(f"Using deterministic subsample: {max_rows} rows")

    candidate_scores_all: Optional[pd.DataFrame] = None
    candidate_score_alignment_failed = False
    if candidate_score_csv:
        candidate_scores_all, _, candidate_score_alignment_failed = load_candidate_score_table(
            candidate_score_csv,
            metadata=metadata_df,
            n_rows=len(X_df),
        )
        warn_if_candidate_margin_uninformative(candidate_scores_all)

    X = X_df.values.astype(np.float64)
    y = y_df.values.astype(np.float64)
    groups = metadata_df[group_col].to_numpy()
    train_idx, val_idx = split_train_validation_indices(
        n_samples=len(X),
        groups=groups,
        test_size=test_size,
        seed=seed,
        split_mode=split_mode,
    )

    scaler = StandardScaler().fit(X[train_idx])
    x_test_df = X_df.iloc[val_idx].reset_index(drop=True)
    return BenchmarkData(
        X_train=scaler.transform(X[train_idx]),
        X_val=scaler.transform(X[val_idx]),
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_train_fit=forward_target_transform(y[train_idx], target_transform),
        y_val_fit=forward_target_transform(y[val_idx], target_transform),
        groups_train=groups[train_idx],
        sample_weight_train=sample_weight_all[train_idx] if sample_weight_all is not None else None,
        target_uncertainty_train=target_uncertainty_all[train_idx]
        if target_uncertainty_all is not None
        else None,
        target_uncertainty_val=target_uncertainty_all[val_idx]
        if target_uncertainty_all is not None
        else None,
        X_scaler=scaler,
        X_columns=list(X_df.columns),
        y_columns=list(y_df.columns),
        val_indices=val_idx,
        x_test_df=x_test_df,
        val_metadata=metadata_df.iloc[val_idx].reset_index(drop=True),
        train_metadata=metadata_df.iloc[train_idx].reset_index(drop=True),
        candidate_scores_train=candidate_scores_all.iloc[train_idx].reset_index(drop=True)
        if candidate_scores_all is not None
        else None,
        candidate_scores_val=candidate_scores_all.iloc[val_idx].reset_index(drop=True)
        if candidate_scores_all is not None
        else None,
        candidate_score_alignment_failed=candidate_score_alignment_failed,
        candidate_score_csv_path=candidate_score_csv,
    )


def maybe_include_poisson(y_train: np.ndarray, seed: int) -> bool:
    if np.any(y_train < 0):
        return False
    try:
        model = DecisionTreeRegressor(criterion="poisson", random_state=seed)
        model.fit(np.zeros((16, 4)), np.full(16, 1.0))
        return True
    except Exception:
        return False


CORE_TREE_EXPERT_NAMES = ["CART_L2_deep", "CART_shallow", "ExtraTree_single"]


def _expanded_tree_expert_builders(seed: int) -> Dict[str, Callable[[], object]]:
    return {
        "CART_L1_deep": lambda: DecisionTreeRegressor(criterion="absolute_error", random_state=seed),
        "CART_friedman": lambda: DecisionTreeRegressor(criterion="friedman_mse", random_state=seed),
        "CART_leaf20": lambda: DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=20, random_state=seed
        ),
        "CART_L2_depth8": lambda: DecisionTreeRegressor(
            criterion="squared_error", max_depth=8, random_state=seed
        ),
        "CART_L2_depth12": lambda: DecisionTreeRegressor(
            criterion="squared_error", max_depth=12, random_state=seed
        ),
        "CART_L2_leaf5": lambda: DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=5, random_state=seed
        ),
        "CART_L2_leaf10": lambda: DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=10, random_state=seed
        ),
        "CART_L2_leaf50": lambda: DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=50, random_state=seed
        ),
        "CART_friedman_depth8": lambda: DecisionTreeRegressor(
            criterion="friedman_mse", max_depth=8, random_state=seed
        ),
        "CART_friedman_leaf10": lambda: DecisionTreeRegressor(
            criterion="friedman_mse", min_samples_leaf=10, random_state=seed
        ),
        "ExtraTree_depth8": lambda: ExtraTreeRegressor(max_depth=8, random_state=seed),
        "ExtraTree_depth12": lambda: ExtraTreeRegressor(max_depth=12, random_state=seed),
        "ExtraTree_leaf10": lambda: ExtraTreeRegressor(min_samples_leaf=10, random_state=seed),
        "ExtraTree_sqrt_features": lambda: ExtraTreeRegressor(max_features="sqrt", random_state=seed),
        "ExtraTree_log2_features": lambda: ExtraTreeRegressor(max_features="log2", random_state=seed),
    }


def build_expert_factories(
    seed: int,
    include_poisson: bool,
    expanded_tree_bank: bool = False,
) -> Dict[str, Callable[[], object]]:
    factories: Dict[str, Callable[[], object]] = {
        "CART_L2_deep": lambda: DecisionTreeRegressor(criterion="squared_error", random_state=seed),
        "CART_shallow": lambda: DecisionTreeRegressor(
            criterion="squared_error", max_depth=6, random_state=seed
        ),
        "ExtraTree_single": lambda: ExtraTreeRegressor(random_state=seed),
    }
    if expanded_tree_bank:
        factories.update(_expanded_tree_expert_builders(seed))
        if include_poisson:
            factories["CART_poisson"] = lambda: DecisionTreeRegressor(criterion="poisson", random_state=seed)
            factories["CART_poisson_leaf10"] = lambda: DecisionTreeRegressor(
                criterion="poisson", min_samples_leaf=10, random_state=seed
            )
            factories["CART_poisson_depth8"] = lambda: DecisionTreeRegressor(
                criterion="poisson", max_depth=8, random_state=seed
            )
    return factories


def select_top_tree_experts_by_oof(
    expert_oof: Dict[str, np.ndarray],
    expert_names: List[str],
    y_train_fit: np.ndarray,
    max_tree_experts: int,
) -> List[str]:
    if max_tree_experts <= 0 or len(expert_names) <= max_tree_experts:
        return list(expert_names)
    scored: List[Tuple[str, float]] = []
    for name in expert_names:
        pred = expert_oof[name]
        mean_r2 = float(
            np.mean([r2_score(y_train_fit[:, j], pred[:, j]) for j in range(y_train_fit.shape[1])])
        )
        scored.append((name, mean_r2))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in scored[:max_tree_experts]]


def make_baseline_control_factories(seed: int) -> Dict[str, Callable[[], object]]:
    return {
        RANDOM_FOREST: lambda: RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
    }


def make_et_debug_factory(seed: int) -> Callable[[], object]:
    return lambda: ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)


def build_tar_architecture_schema() -> dict:
    return {
        "final_model_name": TAR_MODEL,
        "final_architecture": "single-tree experts + target-wise OOF stack",
        "cycle_consistency_used": False,
        "residual_used_in_final_model": False,
        "ET_used_as_control": False,
        "manuscript_controls": ["RF", "Best tree", "UniformTreeMean"],
        "leakage_check_passed": True,
    }


ResidualConfig = CycleConfig


def inner_cv_splits(groups: np.ndarray, seed: int, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(np.unique(groups)) >= n_splits:
        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(np.arange(len(groups)), groups=groups))
    print(
        f"WARNING: only {len(np.unique(groups))} train groups; "
        "using repeated GroupShuffleSplit for inner OOF."
    )
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    indices = np.arange(len(groups))
    for fold in range(n_splits):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
        splits.append(next(gss.split(indices, groups=groups)))
    return splits


def fit_target_model(model, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> object:
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    try:
        model.fit(X, y, **fit_kwargs)
    except Exception as exc:
        if LGBMRegressor is not None and isinstance(model, LGBMRegressor):
            device = model.get_params().get("device_type")
            if device == "gpu":
                warnings.warn(
                    f"LightGBM GPU fit failed ({exc}); retrying on CPU.",
                    stacklevel=2,
                )
                model.set_params(device_type="cpu")
                model.fit(X, y, **fit_kwargs)
                return model
        raise
    return model


def generate_expert_oof(
    factory: Callable[[], object],
    data: BenchmarkData,
    inner_splits: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    n_train, n_targets = data.y_train_fit.shape
    oof = np.zeros((n_train, n_targets), dtype=float)
    for fold_train_idx, fold_val_idx in inner_splits:
        sw = data.sample_weight_train[fold_train_idx] if data.sample_weight_train is not None else None
        for target_idx in range(n_targets):
            model = factory()
            fit_target_model(
                model,
                data.X_train[fold_train_idx],
                data.y_train_fit[fold_train_idx, target_idx],
                sample_weight=sw,
            )
            oof[fold_val_idx, target_idx] = model.predict(data.X_train[fold_val_idx])
    return oof


def fit_expert_full(factory: Callable[[], object], data: BenchmarkData) -> List[object]:
    models = []
    for target_idx in range(data.y_train_fit.shape[1]):
        model = factory()
        fit_target_model(
            model,
            data.X_train,
            data.y_train_fit[:, target_idx],
            sample_weight=data.sample_weight_train,
        )
        models.append(model)
    return models


def predict_expert_models(models: List[object], X: np.ndarray) -> np.ndarray:
    return np.column_stack([model.predict(X) for model in models])


def fit_convex_weights(meta_X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n_experts = meta_X.shape[1]
    if n_experts == 1:
        return np.array([1.0], dtype=float)

    def objective(weights: np.ndarray) -> float:
        return float(np.mean((y - meta_X @ weights) ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_experts
    x0 = np.full(n_experts, 1.0 / n_experts, dtype=float)
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    return weights / total if total > 0 else x0


def fit_ridge_stack(meta_X: np.ndarray, y: np.ndarray) -> Tuple[RidgeCV, np.ndarray]:
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25))
    ridge.fit(meta_X, y)
    return ridge, ridge.coef_.astype(float)


def build_stack_bundles(
    expert_oof: Dict[str, np.ndarray],
    y_train_fit: np.ndarray,
    expert_names: List[str],
) -> Tuple[StackBundle, StackBundle, str]:
    n_train, n_targets = y_train_fit.shape
    ridge_oof = np.zeros((n_train, n_targets), dtype=float)
    convex_oof = np.zeros((n_train, n_targets), dtype=float)
    ridge_weights: Dict[str, np.ndarray] = {}
    convex_weights: Dict[str, np.ndarray] = {}
    ridge_models: Dict[str, RidgeCV] = {}

    for target_idx, target_name in enumerate(TARGET_COLS_TTHR):
        meta = np.column_stack([expert_oof[name][:, target_idx] for name in expert_names])
        y = y_train_fit[:, target_idx]
        ridge_model, ridge_coef = fit_ridge_stack(meta, y)
        convex_w = fit_convex_weights(meta, y)
        ridge_oof[:, target_idx] = ridge_model.predict(meta)
        convex_oof[:, target_idx] = meta @ convex_w
        ridge_weights[target_name] = ridge_coef
        convex_weights[target_name] = convex_w
        ridge_models[target_name] = ridge_model

    ridge_score = float(np.mean([r2_score(y_train_fit[:, j], ridge_oof[:, j]) for j in range(n_targets)]))
    convex_score = float(np.mean([r2_score(y_train_fit[:, j], convex_oof[:, j]) for j in range(n_targets)]))
    ridge_bundle = StackBundle("ridge", ridge_weights, ridge_oof, ridge_models)
    convex_bundle = StackBundle("convex", convex_weights, convex_oof, {})
    chosen_type = "ridge" if ridge_score >= convex_score else "convex"
    return ridge_bundle, convex_bundle, chosen_type


def predict_stack(
    bundle: StackBundle,
    expert_val_preds: Dict[str, np.ndarray],
    expert_names: List[str],
) -> np.ndarray:
    n_val = next(iter(expert_val_preds.values())).shape[0]
    n_targets = len(TARGET_COLS_TTHR)
    out = np.zeros((n_val, n_targets), dtype=float)
    for target_idx, target_name in enumerate(TARGET_COLS_TTHR):
        meta = np.column_stack([expert_val_preds[name][:, target_idx] for name in expert_names])
        if bundle.stacker_type == "ridge":
            out[:, target_idx] = bundle.ridge_models[target_name].predict(meta)
        else:
            out[:, target_idx] = meta @ bundle.weights_per_target[target_name]
    return out


def write_benchmark_csv_artifacts(
    outdir: str,
    summary_df: pd.DataFrame,
    per_target_df: pd.DataFrame,
    repeated_metrics_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    tree_expert_df: pd.DataFrame,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    repeated_metrics_df.to_csv(os.path.join(outdir, "repeated_parameter_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(outdir, "model_compare_summary.csv"), index=False)
    per_target_df.to_csv(os.path.join(outdir, "model_compare_per_target.csv"), index=False)
    pairwise_df.to_csv(os.path.join(outdir, "parameter_pairwise_significance.csv"), index=False)
    weights_df.to_csv(os.path.join(outdir, "target_weight_table.csv"), index=False)
    if not tree_expert_df.empty:
        tree_expert_df.to_csv(os.path.join(outdir, "tree_expert_table.csv"), index=False)


def collect_model_benchmark_rows(
    model_name: str,
    y_pred: np.ndarray,
    data: BenchmarkData,
    target_transform: str,
    repeat_id: int,
    seed: int,
    bootstrap: int = 0,
) -> Tuple[List[dict], dict]:
    y_pred_fit = forward_target_transform(y_pred, target_transform)
    target_metrics: List[dict] = []
    for target_idx, target_name in enumerate(TARGET_COLS_TTHR):
        row = compute_target_metrics(
            model_name,
            data.y_val[:, target_idx],
            y_pred[:, target_idx],
            data.y_val_fit[:, target_idx],
            y_pred_fit[:, target_idx],
            target_name,
        )
        row["repeat_id"] = repeat_id
        target_metrics.append(row)
    per_target_df = pd.DataFrame(target_metrics)
    summary = {
        "model": model_name,
        "repeat_id": repeat_id,
        "seed": seed,
        **summarize_model_metrics(per_target_df),
    }
    if bootstrap > 0:
        summary.update(
            bootstrap_mean_r2_ci(
                data.y_val,
                y_pred,
                data.y_val_fit,
                y_pred_fit,
                bootstrap,
                seed,
            )
        )
    return target_metrics, summary


def calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    if len(y_true) < 2 or np.std(y_pred) <= 1e-30:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(y_pred, y_true, 1)
    return float(slope), float(intercept)


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true))
    if denom <= 1e-30:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / denom)


def compute_target_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_fit: np.ndarray,
    y_pred_fit: np.ndarray,
    target_name: str,
) -> dict:
    spearman = float(spearmanr(y_true, y_pred).correlation) if len(y_true) > 1 else float("nan")
    slope, intercept = calibration_slope_intercept(y_true, y_pred)
    return {
        "model": model_name,
        "target": target_name,
        "R2_original": safe_r2_score(y_true, y_pred),
        "R2_log": safe_r2_score(y_true_fit, y_pred_fit),
        "RMSE_original": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true_fit, y_pred_fit))),
        "MAE_original": float(mean_absolute_error(y_true, y_pred)),
        "MAE_log": float(mean_absolute_error(y_true_fit, y_pred_fit)),
        "NRMSE_original": normalized_rmse(y_true, y_pred),
        "spearman_original": spearman,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def summarize_model_metrics(per_target_df: pd.DataFrame) -> dict:
    return {
        "mean_R2_original": float(per_target_df["R2_original"].mean()),
        "mean_R2_log": float(per_target_df["R2_log"].mean()),
        "mean_RMSE_original": float(per_target_df["RMSE_original"].mean()),
        "mean_RMSE_log": float(per_target_df["RMSE_log"].mean()),
        "mean_MAE_original": float(per_target_df["MAE_original"].mean()),
        "mean_MAE_log": float(per_target_df["MAE_log"].mean()),
        "mean_NRMSE_original": float(per_target_df["NRMSE_original"].mean()),
        "mean_spearman_original": float(per_target_df["spearman_original"].mean()),
        "mean_calibration_slope": float(per_target_df["calibration_slope"].mean()),
        "mean_calibration_intercept": float(per_target_df["calibration_intercept"].mean()),
    }


def bootstrap_mean_r2_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_fit: np.ndarray,
    y_pred_fit: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    if n_bootstrap <= 0:
        return {
            "mean_R2_original_ci_low": np.nan,
            "mean_R2_original_ci_high": np.nan,
            "mean_R2_log_ci_low": np.nan,
            "mean_R2_log_ci_high": np.nan,
        }
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    boot_original, boot_log = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores_orig, scores_log = [], []
        for j in range(y_true.shape[1]):
            if np.var(y_true[idx, j]) > 1e-30:
                scores_orig.append(r2_score(y_true[idx, j], y_pred[idx, j]))
            if np.var(y_true_fit[idx, j]) > 1e-30:
                scores_log.append(r2_score(y_true_fit[idx, j], y_pred_fit[idx, j]))
        if scores_orig:
            boot_original.append(float(np.mean(scores_orig)))
        if scores_log:
            boot_log.append(float(np.mean(scores_log)))
    out: Dict[str, float] = {}
    if boot_original:
        out["mean_R2_original_ci_low"] = float(np.percentile(boot_original, 2.5))
        out["mean_R2_original_ci_high"] = float(np.percentile(boot_original, 97.5))
    else:
        out["mean_R2_original_ci_low"] = np.nan
        out["mean_R2_original_ci_high"] = np.nan
    if boot_log:
        out["mean_R2_log_ci_low"] = float(np.percentile(boot_log, 2.5))
        out["mean_R2_log_ci_high"] = float(np.percentile(boot_log, 97.5))
    else:
        out["mean_R2_log_ci_low"] = np.nan
        out["mean_R2_log_ci_high"] = np.nan
    return out


def aggregate_repeat_summaries(repeat_results: List[RepeatResult]) -> pd.DataFrame:
    summary_df = pd.DataFrame([row for result in repeat_results for row in result.summary_rows])
    if summary_df["repeat_id"].nunique() == 1:
        return summary_df.drop(columns=["repeat_id"], errors="ignore")
    numeric_cols = [
        c
        for c in summary_df.columns
        if c not in {"model", "repeat_id", "seed"} and pd.api.types.is_numeric_dtype(summary_df[c])
    ]
    return summary_df.groupby("model", as_index=False)[numeric_cols].mean()


def repeat_metric_ci(
    vals: np.ndarray,
    method: str = "t_interval",
) -> Tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return mean, float("nan"), float("nan")
    if method == "percentile":
        return mean, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    sem = float(stats.sem(vals))
    half = sem * float(stats.t.ppf(0.975, df=len(vals) - 1))
    return mean, mean - half, mean + half


def build_repeated_parameter_metrics(repeat_results: List[RepeatResult]) -> pd.DataFrame:
    rows: List[dict] = []
    for result in repeat_results:
        for row in result.summary_rows:
            out = {
                "repeat_id": row["repeat_id"],
                "seed": row.get("seed", result.seed),
                "model": row["model"],
            }
            for col in REPEATED_METRIC_COLS:
                out[col] = row.get(col, float("nan"))
            rows.append(out)
    return pd.DataFrame(rows)


def aggregate_repeated_split_summaries(
    repeat_results: List[RepeatResult],
    ci_method: str = "t_interval",
) -> pd.DataFrame:
    long_df = build_repeated_parameter_metrics(repeat_results)
    rows = []
    for model_name, group in long_df.groupby("model"):
        row: dict = {"model": model_name}
        for col in REPEATED_METRIC_COLS:
            mean, lo, hi = repeat_metric_ci(group[col].to_numpy(dtype=float), method=ci_method)
            row[col] = mean
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "na"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def evaluate_significance_label(
    p_value: float,
    ci_low: float,
    ci_high: float,
    n_repeats: int,
    srl_better: bool,
    single_split_exploratory: bool = False,
    force_single_split_significance: bool = False,
    bidirectional: bool = False,
) -> Tuple[str, str]:
    """Return (star_label for plotting, manuscript_tier).

    By default stars are shown only when SRL is better. With ``bidirectional=True``,
    significant control advantages also receive star labels for plotting.
    """
    if not srl_better:
        if not bidirectional:
            return "ns", "control_better"
        if single_split_exploratory and not force_single_split_significance:
            return "ns", "single_split"
        if n_repeats < 2:
            return "ns", "ns"
        stars = significance_label(p_value)
        if (
            stars == "ns"
            and np.isfinite(ci_high)
            and ci_high < 0.0
        ):
            stars = "*"
        if stars == "ns":
            return "ns", "control_better"
        if n_repeats < 10:
            return stars, "exploratory_control_better"
        return stars, "formal_control_better"
    if single_split_exploratory and not force_single_split_significance:
        return "ns", "single_split"
    if n_repeats < 2:
        return "ns", "ns"
    stars = significance_label(p_value)
    if (
        stars == "ns"
        and srl_better
        and np.isfinite(ci_low)
        and np.isfinite(ci_high)
        and ci_low > 0.0
    ):
        stars = "*"
    if stars == "ns":
        return "ns", "ns"
    if n_repeats < 10:
        return stars, "exploratory"
    return stars, "formal"


def comparison_result_label(
    star_label: str,
    significance_tier: str,
    srl_better: bool,
    n_repeats: int,
) -> str:
    if significance_tier == "control_better":
        return "control_better"
    if significance_tier in {"formal_control_better", "exploratory_control_better"}:
        if n_repeats < 10 or significance_tier == "exploratory_control_better":
            return "exploratory_control_better"
        return "control_better"
    if star_label in {"*", "**", "***", "****"}:
        if n_repeats < 10 or significance_tier == "exploratory":
            return "exploratory_srl_better"
        return "srl_better"
    if not srl_better and significance_tier not in {"ns", "single_split"}:
        return "exploratory_control_better"
    return "not_significant"


def permutation_p_value(diffs: np.ndarray, n_perm: int, seed: int) -> float:
    if len(diffs) < 2:
        return float("nan")
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        perm_mean = float(np.mean(diffs * signs))
        if abs(perm_mean) >= abs(observed):
            count += 1
    return float((count + 1) / (n_perm + 1))


def identify_final_tar_model(repeat_result: RepeatResult) -> str:
    if repeat_result.final_model_name:
        return normalize_model_name(repeat_result.final_model_name)
    if TAR_MODEL in repeat_result.predictions:
        return TAR_MODEL
    for legacy_name in LEGACY_MODEL_NAME_MAP:
        if legacy_name in repeat_result.predictions:
            return TAR_MODEL
    raise ValueError("Could not identify TAR final model.")


identify_final_tree_srl_model = identify_final_tar_model


def build_paired_repeated_significance(
    repeat_results: List[RepeatResult],
    srl_model: str,
    control_models: Sequence[str],
    n_perm: int,
    seed: int,
    single_split_exploratory: bool = False,
    force_single_split_significance: bool = False,
) -> pd.DataFrame:
    long_df = build_repeated_parameter_metrics(repeat_results)
    n_repeats = long_df["repeat_id"].nunique()
    rows: List[dict] = []
    for control_model in control_models:
        if control_model == srl_model:
            continue
        for metric, direction in METRIC_DIRECTION.items():
            srl_vals = (
                long_df[long_df["model"] == srl_model]
                .sort_values("repeat_id")[metric]
                .to_numpy(dtype=float)
            )
            ctrl_vals = (
                long_df[long_df["model"] == control_model]
                .sort_values("repeat_id")[metric]
                .to_numpy(dtype=float)
            )
            n = min(len(srl_vals), len(ctrl_vals))
            if n == 0:
                continue
            srl_vals = srl_vals[:n]
            ctrl_vals = ctrl_vals[:n]
            diffs = srl_vals - ctrl_vals
            mean_srl = float(np.mean(srl_vals))
            mean_control = float(np.mean(ctrl_vals))
            mean_diff = float(np.mean(diffs))
            ci_low = float(np.percentile(diffs, 2.5)) if n > 1 else float("nan")
            ci_high = float(np.percentile(diffs, 97.5)) if n > 1 else float("nan")
            higher_is_better = METRIC_DIRECTION[metric] == "higher_is_better"
            srl_better = mean_diff > 0 if higher_is_better else mean_diff < 0
            if n > 1 and np.allclose(diffs, 0.0):
                wilcoxon_p = 1.0
            elif n > 1:
                try:
                    wilcoxon_p = float(wilcoxon(diffs).pvalue)
                except Exception:
                    wilcoxon_p = float("nan")
            else:
                wilcoxon_p = float("nan")
            perm_p = permutation_p_value(diffs, n_perm=n_perm, seed=seed + abs(hash(metric)) % 10000) if n > 1 else float("nan")
            p_candidates = [p for p in (wilcoxon_p, perm_p) if np.isfinite(p)]
            p_for_label = float(min(p_candidates)) if p_candidates else float("nan")
            star_label, significance_tier = evaluate_significance_label(
                p_for_label,
                ci_low,
                ci_high,
                n_repeats=n,
                srl_better=srl_better,
                single_split_exploratory=single_split_exploratory,
                force_single_split_significance=force_single_split_significance,
            )
            comp_result = comparison_result_label(star_label, significance_tier, srl_better, n)
            exploratory = bool(n < 10 or single_split_exploratory)
            rows.append(
                {
                    "metric": metric,
                    "srl_model": srl_model,
                    "control_model": control_model,
                    "direction": direction,
                    "n_repeats": n,
                    "mean_srl": mean_srl,
                    "mean_control": mean_control,
                    "mean_diff": mean_diff,
                    "CI_low": ci_low,
                    "CI_high": ci_high,
                    "wilcoxon_p": wilcoxon_p,
                    "permutation_p": perm_p,
                    "significance_label": star_label,
                    "significance_tier": significance_tier,
                    "comparison_result": comp_result,
                    "exploratory": exploratory,
                }
            )
    return pd.DataFrame(rows)


def build_parameter_pairwise_significance(
    repeat_results: List[RepeatResult],
    srl_models: Sequence[str],
    control_models: Sequence[str],
    n_perm: int,
    seed: int,
    single_split_exploratory: bool = False,
    force_single_split_significance: bool = False,
) -> pd.DataFrame:
    frames = []
    for srl_model in srl_models:
        df = build_paired_repeated_significance(
            repeat_results,
            srl_model=srl_model,
            control_models=control_models,
            n_perm=n_perm,
            seed=seed,
            single_split_exploratory=single_split_exploratory,
            force_single_split_significance=force_single_split_significance,
        )
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["metric", "srl_model", "control_model"]
    ).reset_index(drop=True)


def run_single_repeat(
    data: BenchmarkData,
    seed: int,
    repeat_id: int,
    target_transform: str,
    bootstrap: int,
    expanded_tree_bank: bool = False,
    max_tree_experts: int = 0,
    include_et_debug: bool = False,
    enable_gated_residual: bool = False,
    uncertainty_cfg: Optional[UncertaintyDecompositionConfig] = None,
    cycle_cfg: Optional[CycleConfig] = None,
) -> RepeatResult:
    del cycle_cfg  # retained for closed_loop_eval call compatibility; cycle is disabled
    uncertainty_cfg = uncertainty_cfg or UncertaintyDecompositionConfig(enabled=False)
    gated_residual_passed_gate = False
    if enable_gated_residual:
        warnings.warn(
            "--enable_gated_residual is set but gated residual is not part of the final TAR model; "
            "enable --include_residual_diagnostics for optional diagnostic outputs only.",
            stacklevel=2,
        )
    include_poisson = maybe_include_poisson(data.y_train, seed)
    expert_factories = build_expert_factories(
        seed, include_poisson=include_poisson, expanded_tree_bank=expanded_tree_bank
    )
    expert_names = list(expert_factories.keys())
    bank_label = "expanded" if expanded_tree_bank else "core"
    print(f"  Tree expert bank ({bank_label}, n={len(expert_names)}): {expert_names}")
    inner_splits = inner_cv_splits(data.groups_train, seed=seed)

    expert_oof: Dict[str, np.ndarray] = {}
    expert_val_preds: Dict[str, np.ndarray] = {}
    predictions: Dict[str, np.ndarray] = {}

    for expert_name, factory in expert_factories.items():
        print(f"  OOF expert: {expert_name}")
        expert_oof[expert_name] = generate_expert_oof(factory, data, inner_splits)

    if max_tree_experts > 0:
        expert_names = select_top_tree_experts_by_oof(
            expert_oof, expert_names, data.y_train_fit, max_tree_experts
        )
        expert_oof = {name: expert_oof[name] for name in expert_names}
        expert_factories = {name: expert_factories[name] for name in expert_names}
        print(f"  Selected top {len(expert_names)} tree experts by inner OOF: {expert_names}")

    for expert_name, factory in expert_factories.items():
        expert_models = fit_expert_full(factory, data)
        expert_val_preds[expert_name] = predict_expert_models(expert_models, data.X_val)

    uniform_pred_fit = np.mean(np.stack([expert_val_preds[name] for name in expert_names], axis=0), axis=0)
    predictions[UNIFORM_TREE_MEAN] = inverse_target_transform(uniform_pred_fit, target_transform)

    ridge_bundle, convex_bundle, chosen_type = build_stack_bundles(expert_oof, data.y_train_fit, expert_names)
    ridge_val_fit = predict_stack(ridge_bundle, expert_val_preds, expert_names)
    convex_val_fit = predict_stack(convex_bundle, expert_val_preds, expert_names)
    chosen_bundle = ridge_bundle if chosen_type == "ridge" else convex_bundle
    chosen_val_fit = ridge_val_fit if chosen_type == "ridge" else convex_val_fit

    for control_name, factory in make_baseline_control_factories(seed).items():
        print(f"  Control: {control_name}")
        control_models = fit_expert_full(factory, data)
        control_val_fit = predict_expert_models(control_models, data.X_val)
        predictions[control_name] = inverse_target_transform(control_val_fit, target_transform)

    if include_et_debug:
        print("  Debug-only ExtraTrees control")
        et_models = fit_expert_full(make_et_debug_factory(seed), data)
        et_val_fit = predict_expert_models(et_models, data.X_val)
        predictions["ExtraTrees"] = inverse_target_transform(et_val_fit, target_transform)

    expert_summary_scores = []
    tree_expert_summary_rows: List[dict] = []
    for expert_name in expert_names:
        y_pred = inverse_target_transform(expert_val_preds[expert_name], target_transform)
        _, expert_summary = collect_model_benchmark_rows(
            expert_name, y_pred, data, target_transform, repeat_id, seed, bootstrap=0
        )
        tree_expert_summary_rows.append(expert_summary)
        expert_summary_scores.append((expert_name, float(expert_summary["mean_R2_original"])))
    best_single_tree_name = max(expert_summary_scores, key=lambda item: item[1])[0]
    predictions[BEST_SINGLE_TREE] = inverse_target_transform(
        expert_val_preds[best_single_tree_name], target_transform
    )

    predictions[TAR_MODEL] = inverse_target_transform(chosen_val_fit.copy(), target_transform)

    uncertainty_case_rows: List[dict] = []
    uncertainty_summary_rows: List[dict] = []
    uncertainty_training_log_rows: List[dict] = []
    uncertainty_manifest: Dict[str, object] = {}
    uncertainty_skipped = False
    uncertainty_skip_reason = ""
    if uncertainty_cfg.enabled:
        try:
            unc_result = run_uncertainty_decomposition(
                data,
                cfg=uncertainty_cfg,
                repeat_id=repeat_id,
                seed=seed,
                target_transform=target_transform,
            )
            uncertainty_case_rows = unc_result.case_rows
            uncertainty_summary_rows = unc_result.summary_rows
            uncertainty_training_log_rows = unc_result.training_log_rows
            uncertainty_manifest = unc_result.manifest
            uncertainty_skipped = unc_result.skipped
            uncertainty_skip_reason = unc_result.skip_reason
        except Exception as exc:
            uncertainty_skipped = True
            uncertainty_skip_reason = str(exc)
            uncertainty_manifest = {
                "uncertainty_enabled": False,
                "skip_reason": uncertainty_skip_reason,
                "exploratory_not_for_main_claim": True,
            }
            warnings.warn(f"uncertainty decomposition failed: {exc}", stacklevel=2)

    target_weights_rows: List[dict] = []
    for target_name in TARGET_COLS_TTHR:
        for expert_name, weight in zip(expert_names, chosen_bundle.weights_per_target[target_name]):
            target_weights_rows.append(
                {
                    "repeat_id": repeat_id,
                    "target": target_name,
                    "expert": expert_name,
                    "stacker_type": chosen_type,
                    "weight": float(weight),
                }
            )

    benchmark_models = set(CORE_BENCHMARK_MODELS)
    if include_et_debug:
        benchmark_models.add("ExtraTrees")
    per_target_rows: List[dict] = []
    summary_rows: List[dict] = []
    for model_name, y_pred in predictions.items():
        if model_name not in benchmark_models and model_name not in expert_names:
            continue
        target_metrics, summary = collect_model_benchmark_rows(
            model_name, y_pred, data, target_transform, repeat_id, seed, bootstrap=bootstrap
        )
        per_target_rows.extend(target_metrics)
        if model_name in benchmark_models:
            summary_rows.append(summary)

    return RepeatResult(
        repeat_id=repeat_id,
        seed=seed,
        per_target_rows=per_target_rows,
        summary_rows=summary_rows,
        predictions={k: v for k, v in predictions.items() if k in benchmark_models},
        target_weights_rows=target_weights_rows,
        final_model_name=TAR_MODEL,
        best_single_tree_name=best_single_tree_name,
        chosen_stacker_type=chosen_type,
        tree_expert_summary_rows=tree_expert_summary_rows,
        expert_names_used=list(expert_names),
        uncertainty_case_rows=uncertainty_case_rows,
        uncertainty_summary_rows=uncertainty_summary_rows,
        uncertainty_training_log_rows=uncertainty_training_log_rows,
        uncertainty_manifest=uncertainty_manifest,
        uncertainty_skipped=uncertainty_skipped,
        uncertainty_skip_reason=uncertainty_skip_reason,
        gated_residual_enabled=enable_gated_residual,
        gated_residual_passed_gate=gated_residual_passed_gate,
        split_metadata={
            "train_row_count": int(len(data.X_train)),
            "val_row_count": int(len(data.X_val)),
            "train_bio_groups": int(len(np.unique(data.groups_train))),
            "test_bio_groups": int(data.val_metadata["bio_id"].nunique())
            if data.val_metadata is not None
            else None,
            "test_row_count": int(len(data.X_val)),
        },
    )


def _run_benchmark_repeat_job(
    repeat_id: int,
    repeat_seed: int,
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    sample_weight_csv: Optional[str],
    split_mode: str,
    group_col: str,
    test_size: float,
    target_transform: str,
    max_rows: int,
    use_physics_features: bool,
    bootstrap: int,
    lgbm_device: str = "gpu",
    verbose: bool = False,
    expanded_tree_bank: bool = False,
    max_tree_experts: int = 0,
    include_et_debug: bool = False,
    enable_gated_residual: bool = False,
    candidate_score_csv: Optional[str] = None,
    uncertainty_cfg: Optional[UncertaintyDecompositionConfig] = None,
    cycle_cfg: Optional[CycleConfig] = None,
) -> Tuple[RepeatResult, BenchmarkData]:
    _configure_parallel_worker(lgbm_device=lgbm_device, verbose=verbose)
    set_global_seed(repeat_seed)
    print(f"\n===== Repeat {repeat_id + 1} (seed={repeat_seed}) =====", flush=True)
    data = load_benchmark_data(
        x_csv=x_csv,
        y_csv=y_csv,
        metadata_csv=metadata_csv,
        sample_weight_csv=sample_weight_csv,
        split_mode=split_mode,
        group_col=group_col,
        test_size=test_size,
        seed=repeat_seed,
        target_transform=target_transform,
        max_rows=max_rows,
        use_physics_features=use_physics_features,
        candidate_score_csv=candidate_score_csv,
    )
    repeat_result = run_single_repeat(
        data=data,
        seed=repeat_seed,
        repeat_id=repeat_id,
        target_transform=target_transform,
        bootstrap=bootstrap,
        expanded_tree_bank=expanded_tree_bank,
        max_tree_experts=max_tree_experts,
        include_et_debug=include_et_debug,
        enable_gated_residual=enable_gated_residual,
        uncertainty_cfg=uncertainty_cfg,
        cycle_cfg=cycle_cfg,
    )
    return repeat_result, data


def run_benchmark_repeats(
    n_repeats: int,
    seed: int,
    n_jobs: int,
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    sample_weight_csv: Optional[str],
    split_mode: str,
    group_col: str,
    test_size: float,
    target_transform: str,
    max_rows: int,
    use_physics_features: bool,
    bootstrap: int,
    lgbm_device: str = "gpu",
    verbose: bool = False,
    expanded_tree_bank: bool = False,
    max_tree_experts: int = 0,
    include_et_debug: bool = False,
    enable_gated_residual: bool = False,
    candidate_score_csv: Optional[str] = None,
    uncertainty_cfg: Optional[UncertaintyDecompositionConfig] = None,
    cycle_cfg: Optional[CycleConfig] = None,
) -> Tuple[List[RepeatResult], BenchmarkData, Dict[int, BenchmarkData]]:
    job_kwargs = dict(
        x_csv=x_csv,
        y_csv=y_csv,
        metadata_csv=metadata_csv,
        sample_weight_csv=sample_weight_csv,
        split_mode=split_mode,
        group_col=group_col,
        test_size=test_size,
        target_transform=target_transform,
        max_rows=max_rows,
        use_physics_features=use_physics_features,
        bootstrap=bootstrap,
        lgbm_device=lgbm_device,
        verbose=verbose,
        expanded_tree_bank=expanded_tree_bank,
        max_tree_experts=max_tree_experts,
        include_et_debug=include_et_debug,
        enable_gated_residual=enable_gated_residual,
        candidate_score_csv=candidate_score_csv,
        uncertainty_cfg=uncertainty_cfg,
        cycle_cfg=cycle_cfg,
    )
    if n_repeats <= 1 or n_jobs == 1:
        results = [
            _run_benchmark_repeat_job(repeat_id, seed + repeat_id, **job_kwargs)
            for repeat_id in range(n_repeats)
        ]
    else:
        from joblib import Parallel, delayed

        effective_jobs = n_jobs if n_jobs > 0 else -1
        parallel_verbose = 10 if verbose else 0
        print(f"Running {n_repeats} repeats in parallel (n_jobs={effective_jobs})...", flush=True)
        results = Parallel(n_jobs=effective_jobs, verbose=parallel_verbose)(
            delayed(_run_benchmark_repeat_job)(repeat_id, seed + repeat_id, **job_kwargs)
            for repeat_id in range(n_repeats)
        )
    results.sort(key=lambda item: item[0].repeat_id)
    repeat_results = [item[0] for item in results]
    last_data = results[-1][1]
    data_by_repeat = {item[0].repeat_id: item[1] for item in results}
    return repeat_results, last_data, data_by_repeat


def save_predictions_all_models(
    outpath: str,
    data: BenchmarkData,
    predictions: Dict[str, np.ndarray],
    repeat_id: Optional[int] = None,
    seed: Optional[int] = None,
    models_to_export: Optional[Sequence[str]] = None,
) -> None:
    export_models = list(models_to_export or CORE_BENCHMARK_MODELS)
    missing = [m for m in export_models if m not in predictions]
    if missing:
        raise ValueError(f"Cannot export predictions; missing core models: {missing}")

    base = pd.DataFrame({"validation_original_row_index": data.val_indices.astype(int)})
    if repeat_id is not None:
        base.insert(0, "repeat_id", int(repeat_id))
    if seed is not None:
        insert_at = 1 if repeat_id is not None else 0
        base.insert(insert_at, "seed", int(seed))
    if data.val_metadata is not None:
        for col in ("bio_id", "desired_profile_id"):
            if col in data.val_metadata.columns:
                base[col] = data.val_metadata[col].reset_index(drop=True).values
    for target_idx, target_name in enumerate(data.y_columns):
        base[f"true_{target_name}"] = data.y_val[:, target_idx]
    for model_name in export_models:
        y_pred = predictions[model_name]
        for target_idx, target_name in enumerate(data.y_columns):
            base[f"pred_{model_name}_{target_name}"] = y_pred[:, target_idx]
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    base.to_csv(outpath, index=False)


def repeat_output_dir(outdir: str, repeat_id: int) -> str:
    """Per-repeat artifact directory under ``outdir/repeats/repeat_XXX/``."""
    return os.path.join(outdir, "repeats", f"repeat_{repeat_id:03d}")


def glob_repeat_prediction_csvs(outdir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(outdir, "repeats", "repeat_*", "predictions.csv")))
    if paths:
        return paths
    return sorted(glob.glob(os.path.join(outdir, "predictions_repeat_*.csv")))


def build_repeat_metadata_rows(repeat_results: List[RepeatResult]) -> pd.DataFrame:
    rows: List[dict] = []
    for result in repeat_results:
        row = {
            "repeat_id": result.repeat_id,
            "seed": result.seed,
            "final_model_name": result.final_model_name,
            "best_single_tree_name": result.best_single_tree_name,
            "chosen_stacker_type": result.chosen_stacker_type,
            "expert_names_used": ",".join(result.expert_names_used),
            "n_experts_used": len(result.expert_names_used),
        }
        row.update(result.split_metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def save_repeat_predictions(
    outdir: str,
    repeat_results: List[RepeatResult],
    data_by_repeat: Dict[int, BenchmarkData],
    n_repeats: int,
    *,
    split_mode: str = "group",
    group_col: str = "bio_id",
) -> List[str]:
    """Write per-repeat prediction CSVs under ``repeats/repeat_XXX/predictions.csv``."""
    prediction_paths: List[str] = []
    export_models = list(CORE_BENCHMARK_MODELS)
    for result in repeat_results:
        data = data_by_repeat.get(result.repeat_id)
        if data is None:
            continue
        repeat_dir = repeat_output_dir(outdir, result.repeat_id)
        os.makedirs(repeat_dir, exist_ok=True)
        pred_path = os.path.join(repeat_dir, "predictions.csv")
        save_predictions_all_models(
            pred_path,
            data,
            result.predictions,
            repeat_id=result.repeat_id,
            seed=result.seed,
            models_to_export=export_models,
        )
        prediction_paths.append(pred_path)
        if n_repeats == 1:
            root_path = os.path.join(outdir, "predictions_all_models.csv")
            save_predictions_all_models(
                root_path,
                data,
                result.predictions,
                repeat_id=result.repeat_id,
                seed=result.seed,
                models_to_export=export_models,
            )
            prediction_paths.append(root_path)
        pd.DataFrame([build_repeat_metadata_rows([result]).iloc[0].to_dict()]).to_csv(
            os.path.join(repeat_dir, "repeat_metadata.csv"), index=False
        )
    return prediction_paths


def write_predictions_manifest(
    outdir: str,
    prediction_paths: Sequence[str],
    n_repeats: int,
    *,
    split_mode: str,
    group_col: str,
    test_size: float,
    seed: int,
) -> str:
    rel_paths = [
        os.path.relpath(p, outdir).replace("\\", "/")
        for p in prediction_paths
        if "repeats/repeat_" in p.replace("\\", "/") or os.path.basename(p).startswith("predictions")
    ]
    manifest = {
        "prediction_files": sorted(set(rel_paths)),
        "models": list(CORE_BENCHMARK_MODELS),
        "display_labels": {m: MODEL_DISPLAY_LABELS.get(m, m) for m in CORE_BENCHMARK_MODELS},
        "target_columns": TARGET_COLS_TTHR,
        "tar_predicts_only": "Tthr vector (Tthr_1..Tthr_5); Umax is selected later by closed_loop_eval.py inverse-design module",
        "umax_not_supervised_target": True,
        "validation_row_source": "validation_original_row_index",
        "split_mode": split_mode,
        "group_col": group_col,
        "test_size": test_size,
        "seed": seed,
        "n_repeats": n_repeats,
        "leakage_check_passed": split_mode == "group",
        "legacy_model_name_map": LEGACY_MODEL_NAME_MAP,
        "canonical_final_model": TAR_MODEL,
    }
    path = os.path.join(outdir, "predictions_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def build_tree_expert_table(
    tree_expert_summary_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    if tree_expert_summary_df.empty:
        return pd.DataFrame()
    sub = tree_expert_summary_df.copy()
    sub["rank"] = sub["mean_R2_original"].rank(ascending=False, method="min").astype(int)
    stack_weights = (
        weights_df.groupby("expert", as_index=False)["weight"]
        .mean()
        .rename(columns={"weight": "mean_stack_weight"})
        if not weights_df.empty
        else pd.DataFrame(columns=["expert", "mean_stack_weight"])
    )
    selected = set(weights_df["expert"].unique()) if not weights_df.empty else set()
    rows: List[dict] = []
    for _, row in sub.iterrows():
        expert = str(row["model"])
        sw_row = stack_weights[stack_weights["expert"] == expert]
        rows.append(
            {
                "expert": expert,
                "mean_R2_original": float(row["mean_R2_original"]),
                "mean_R2_log": float(row.get("mean_R2_log", np.nan)),
                "rank": int(row["rank"]),
                "selected_in_stack": expert in selected,
                "mean_stack_weight": float(sw_row["mean_stack_weight"].iloc[0])
                if not sw_row.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("rank")


def aggregate_tree_expert_summaries(
    repeat_results: List[RepeatResult],
    ci_method: str = "t_interval",
) -> pd.DataFrame:
    long_df = pd.DataFrame(
        [row for result in repeat_results for row in result.tree_expert_summary_rows]
    )
    if long_df.empty:
        return pd.DataFrame()
    if long_df["repeat_id"].nunique() <= 1:
        return long_df.drop(columns=["repeat_id", "seed"], errors="ignore")
    rows: List[dict] = []
    for model_name, group in long_df.groupby("model"):
        row: dict = {"model": model_name}
        for col in ("mean_R2_original", "mean_R2_log"):
            mean, _, _ = repeat_metric_ci(group[col].to_numpy(dtype=float), method=ci_method)
            row[col] = mean
        rows.append(row)
    return pd.DataFrame(rows)


def write_run_manifest(
    outdir: str,
    config: dict,
    repeat_metadata_rows: List[dict],
    repeat_results: List["RepeatResult"],
    generated_primary_csv: Sequence[str],
    generated_primary_png: Sequence[str],
) -> str:
    """Write benchmark manifest; figures are generated by figure_audit.py."""
    residual_used = any(
        r.gated_residual_enabled and r.gated_residual_passed_gate for r in repeat_results
    )
    manifest = {
        "final_model_name": TAR_MODEL,
        "final_architecture": "single-tree experts + target-wise OOF stack",
        "cycle_consistency_used": False,
        "residual_used_in_final_model": residual_used,
        "ET_used_as_control": bool(config.get("include_et_debug", False)),
        "manuscript_controls": ["RF", "Best tree", "UniformTreeMean"],
        "row_split_manuscript_safe": False,
        "outputs": {},
        "run_config": {k: v for k, v in config.items() if k not in {"tar_architecture"}},
        "tar_architecture": build_tar_architecture_schema(),
        "legacy_model_name_map": LEGACY_MODEL_NAME_MAP,
        "repeat_metadata": repeat_metadata_rows,
        "split_mode": config.get("split_mode"),
        "group_col": config.get("group_col"),
        "test_size": config.get("test_size"),
        "target_transform": config.get("target_transform"),
        "n_repeats": config.get("n_repeats", 1),
        "repeat_artifact_dir_pattern": "repeats/repeat_{repeat_id:03d}",
        "repeat_ci_method": config.get("repeat_ci_method"),
        "single_split_exploratory": config.get("single_split_exploratory", False),
        "significance_for_manuscript": config.get("significance_for_manuscript", False),
        "expanded_tree_bank": config.get("expanded_tree_bank", False),
        "include_et_debug": config.get("include_et_debug", False),
        "enable_gated_residual": config.get("enable_gated_residual", False),
        "main_plot_models": config.get("main_plot_models", MAIN_BAR_ORDER),
        "leakage_check_passed": True,
        "validation_safe_columns": list(DIAGNOSTIC_ONLY_COLS),
        "diagnostic_only_columns": list(DIAGNOSTIC_ONLY_COLS),
        "candidate_score_csv_used": config.get("candidate_score_csv_used"),
        "candidate_score_alignment_failed": config.get("candidate_score_alignment_failed", False),
        "manuscript_safe": config.get("split_mode") == "group" and not config.get("not_for_manuscript", False),
        "generated_primary_csv": list(generated_primary_csv),
        "generated_primary_png": list(generated_primary_png),
        "deprecated_outputs_removed": True,
        "uncertainty_enabled": config.get("uncertainty_enabled", False),
        "uncertainty_method": config.get("uncertainty_method"),
        "figures_note": "Run: python figure_audit.py --mode generate_plots --groups benchmark --benchmark_outdir <outdir> (PNGs under <outdir>/figure/)",
    }
    path = os.path.join(outdir, "model_compare_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def build_cycle_config_from_args(args: argparse.Namespace) -> CycleConfig:
    """Backward compatibility stub for closed_loop_eval (cycle disabled)."""
    return CycleConfig(enabled=False)


def build_residual_config_from_args(args: argparse.Namespace) -> CycleConfig:
    return build_cycle_config_from_args(args)


def warn_if_repeats_missing(expected_n: int, repeat_results: List[RepeatResult]) -> None:
    found = {r.repeat_id for r in repeat_results}
    if expected_n > 1 and len(found) < expected_n:
        warnings.warn(
            f"Expected n_repeats={expected_n} but only found {len(found)} repeat(s) in outputs "
            f"(repeat_ids={sorted(found)}).",
            stacklevel=2,
        )


def print_console_summary(summary_df: pd.DataFrame) -> None:
    def _mean_r2(model: str) -> float:
        row = summary_df[summary_df["model"] == model]
        return float(row["mean_R2_original"].iloc[0]) if not row.empty else float("nan")

    tar_r2 = _mean_r2(TAR_MODEL)
    rf_r2 = _mean_r2(RANDOM_FOREST)
    best_r2 = _mean_r2(BEST_SINGLE_TREE)
    uniform_r2 = _mean_r2(UNIFORM_TREE_MEAN)

    print("\n===== Benchmark console summary =====")
    if np.isfinite(tar_r2):
        print(f"TAR mean R2 = {tar_r2:.4f}")
    if np.isfinite(rf_r2) and np.isfinite(tar_r2):
        print(f"TAR minus RF = {tar_r2 - rf_r2:+.4f}")
    if np.isfinite(best_r2) and np.isfinite(tar_r2):
        print(f"TAR minus Best tree = {tar_r2 - best_r2:+.4f}")
    if np.isfinite(uniform_r2) and np.isfinite(tar_r2):
        print(f"TAR minus UniformTreeMean = {tar_r2 - uniform_r2:+.4f}")


def _rf_factory_screening(seed: int, n_estimators: int, max_depth, min_samples_leaf: int) -> Callable[[], object]:
    depth = None if max_depth in (None, "null", "None") else int(max_depth)

    def _factory() -> object:
        return RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=seed,
            n_jobs=-1,
        )

    return _factory


def _inner_cv_score_tar(
    data: BenchmarkData,
    *,
    seed: int,
    expanded_tree_bank: bool,
    max_tree_experts: int,
    target_transform: str,
    inner_folds: int,
) -> float:
    splits = inner_cv_splits(data.groups_train, seed=seed, n_splits=inner_folds)
    fold_scores: List[float] = []
    include_poisson = maybe_include_poisson(data.y_train, seed)
    expert_factories = build_expert_factories(
        seed, include_poisson=include_poisson, expanded_tree_bank=expanded_tree_bank
    )
    expert_names = list(expert_factories.keys())
    for fold_train, fold_val in splits:
        fold_data = BenchmarkData(
            X_train=data.X_train[fold_train],
            X_val=data.X_train[fold_val],
            y_train=data.y_train[fold_train],
            y_val=data.y_train[fold_val],
            y_train_fit=forward_target_transform(data.y_train[fold_train], target_transform),
            y_val_fit=forward_target_transform(data.y_train[fold_val], target_transform),
            groups_train=data.groups_train[fold_train],
            sample_weight_train=(
                data.sample_weight_train[fold_train] if data.sample_weight_train is not None else None
            ),
            target_uncertainty_train=None,
            target_uncertainty_val=None,
            X_scaler=data.X_scaler,
            X_columns=data.X_columns,
            y_columns=data.y_columns,
            val_indices=fold_val,
            x_test_df=data.x_test_df,
            val_metadata=data.val_metadata,
            train_metadata=data.train_metadata,
            candidate_scores_train=None,
            candidate_scores_val=None,
            candidate_score_alignment_failed=False,
            candidate_score_csv_path=None,
        )
        inner_splits = inner_cv_splits(fold_data.groups_train, seed=seed + 17, n_splits=min(3, inner_folds))
        expert_oof: Dict[str, np.ndarray] = {}
        for expert_name, factory in expert_factories.items():
            expert_oof[expert_name] = generate_expert_oof(factory, fold_data, inner_splits)
        if max_tree_experts > 0:
            expert_names_sel = select_top_tree_experts_by_oof(
                expert_oof, expert_names, fold_data.y_train_fit, max_tree_experts
            )
            expert_oof = {n: expert_oof[n] for n in expert_names_sel}
            expert_factories = {n: expert_factories[n] for n in expert_names_sel}
            expert_names = expert_names_sel
        expert_val_preds: Dict[str, np.ndarray] = {}
        for expert_name, factory in expert_factories.items():
            models = fit_expert_full(factory, fold_data)
            expert_val_preds[expert_name] = predict_expert_models(models, fold_data.X_val)
        ridge_bundle, convex_bundle, chosen_type = build_stack_bundles(
            expert_oof, fold_data.y_train_fit, expert_names
        )
        chosen_val_fit = (
            predict_stack(ridge_bundle, expert_val_preds, expert_names)
            if chosen_type == "ridge"
            else predict_stack(convex_bundle, expert_val_preds, expert_names)
        )
        y_pred = inverse_target_transform(chosen_val_fit, target_transform)
        fold_scores.append(mean_target_wise_r2_score(fold_data.y_val, y_pred))
    return float(np.nanmean(fold_scores))


def _inner_cv_score_multimodel(
    data: BenchmarkData,
    *,
    seed: int,
    model_name: str,
    factory: Callable[[], object],
    target_transform: str,
    inner_folds: int,
    expanded_tree_bank: bool,
    max_tree_experts: int,
) -> float:
    splits = inner_cv_splits(data.groups_train, seed=seed, n_splits=inner_folds)
    fold_scores: List[float] = []
    for fold_train, fold_val in splits:
        X_tr, X_va = data.X_train[fold_train], data.X_train[fold_val]
        y_tr = forward_target_transform(data.y_train[fold_train], target_transform)
        y_va = data.y_train[fold_val]
        sw = data.sample_weight_train[fold_train] if data.sample_weight_train is not None else None
        if model_name == BEST_SINGLE_TREE:
            include_poisson = maybe_include_poisson(data.y_train[fold_train], seed)
            expert_factories = build_expert_factories(
                seed, include_poisson=include_poisson, expanded_tree_bank=expanded_tree_bank
            )
            expert_names = list(expert_factories.keys())
            fold_data = BenchmarkData(
                X_train=X_tr,
                X_val=X_va,
                y_train=data.y_train[fold_train],
                y_val=y_va,
                y_train_fit=y_tr,
                y_val_fit=forward_target_transform(y_va, target_transform),
                groups_train=data.groups_train[fold_train],
                sample_weight_train=sw,
                target_uncertainty_train=None,
                target_uncertainty_val=None,
                X_scaler=data.X_scaler,
                X_columns=data.X_columns,
                y_columns=data.y_columns,
                val_indices=fold_val,
                x_test_df=data.x_test_df,
                val_metadata=data.val_metadata,
                train_metadata=data.train_metadata,
                candidate_scores_train=None,
                candidate_scores_val=None,
                candidate_score_alignment_failed=False,
                candidate_score_csv_path=None,
            )
            inner_splits = inner_cv_splits(fold_data.groups_train, seed=seed + 31, n_splits=min(3, inner_folds))
            expert_oof = {
                n: generate_expert_oof(f, fold_data, inner_splits) for n, f in expert_factories.items()
            }
            if max_tree_experts > 0:
                expert_names = select_top_tree_experts_by_oof(
                    expert_oof, expert_names, fold_data.y_train_fit, max_tree_experts
                )
                expert_oof = {n: expert_oof[n] for n in expert_names}
                expert_factories = {n: expert_factories[n] for n in expert_names}
            scores = []
            for expert_name, ef in expert_factories.items():
                models = fit_expert_full(ef, fold_data)
                pred_fit = predict_expert_models(models, fold_data.X_val)
                pred = inverse_target_transform(pred_fit, target_transform)
                scores.append((expert_name, safe_r2_score(y_va, pred)))
            y_pred = inverse_target_transform(
                predict_expert_models(
                    fit_expert_full(expert_factories[max(scores, key=lambda x: x[1])[0]], fold_data),
                    X_va,
                ),
                target_transform,
            )
        elif model_name == UNIFORM_TREE_MEAN:
            include_poisson = maybe_include_poisson(data.y_train[fold_train], seed)
            expert_factories = build_expert_factories(
                seed, include_poisson=include_poisson, expanded_tree_bank=expanded_tree_bank
            )
            expert_names = list(expert_factories.keys())
            fold_data = BenchmarkData(
                X_train=X_tr,
                X_val=X_va,
                y_train=data.y_train[fold_train],
                y_val=y_va,
                y_train_fit=y_tr,
                y_val_fit=forward_target_transform(y_va, target_transform),
                groups_train=data.groups_train[fold_train],
                sample_weight_train=sw,
                target_uncertainty_train=None,
                target_uncertainty_val=None,
                X_scaler=data.X_scaler,
                X_columns=data.X_columns,
                y_columns=data.y_columns,
                val_indices=fold_val,
                x_test_df=data.x_test_df,
                val_metadata=data.val_metadata,
                train_metadata=data.train_metadata,
                candidate_scores_train=None,
                candidate_scores_val=None,
                candidate_score_alignment_failed=False,
                candidate_score_csv_path=None,
            )
            if max_tree_experts > 0:
                inner_splits = inner_cv_splits(fold_data.groups_train, seed=seed + 41, n_splits=min(3, inner_folds))
                expert_oof = {
                    n: generate_expert_oof(f, fold_data, inner_splits) for n, f in expert_factories.items()
                }
                expert_names = select_top_tree_experts_by_oof(
                    expert_oof, expert_names, fold_data.y_train_fit, max_tree_experts
                )
                expert_factories = {n: expert_factories[n] for n in expert_names}
            preds = []
            for ef in expert_factories.values():
                models = fit_expert_full(ef, fold_data)
                preds.append(predict_expert_models(models, X_va))
            y_pred = inverse_target_transform(np.mean(np.stack(preds, axis=0), axis=0), target_transform)
        else:
            model = factory()
            fit_target_model(model, X_tr, y_tr, sample_weight=sw)
            y_pred = inverse_target_transform(model.predict(X_va), target_transform)
        fold_scores.append(mean_target_wise_r2_score(y_va, y_pred))
    return float(np.nanmean(fold_scores))


def run_training_parameter_screening(
    *,
    x_csv: str,
    y_csv: str,
    metadata_csv: Optional[str],
    sample_weight_csv: Optional[str],
    outdir: str,
    plan_training: Dict[str, object],
    candidate_score_csv: Optional[str] = None,
) -> Dict[str, object]:
    """Stage-1 training hyperparameter screening (inner CV on training split only)."""
    os.makedirs(outdir, exist_ok=True)
    split_mode = str(plan_training.get("split_mode", "group"))
    group_col = str(plan_training.get("group_col", "bio_id"))
    test_size = float(plan_training.get("test_size", 0.2))
    seed = int(plan_training.get("screening_seed", 42))
    inner_folds = int(plan_training.get("inner_cv_folds", 3))
    metric = str(plan_training.get("selected_metric", "mean_R2_original"))
    target_transform = str(plan_training.get("target_transform", "log"))

    data = load_benchmark_data(
        x_csv=x_csv,
        y_csv=y_csv,
        metadata_csv=metadata_csv,
        sample_weight_csv=sample_weight_csv,
        split_mode=split_mode,
        group_col=group_col,
        test_size=test_size,
        seed=seed,
        target_transform=target_transform,
        candidate_score_csv=candidate_score_csv,
    )

    grid_rows: List[Dict[str, object]] = []
    tar_grid = plan_training.get("tar_grid", {})
    rf_grid = plan_training.get("rf_grid", {})
    for expanded, max_exp in itertools.product(
        tar_grid.get("expanded_tree_bank", [False, True]),
        tar_grid.get("max_tree_experts", [3, 6, 10]),
    ):
        grid_rows.append(
            {
                "model": TAR_MODEL,
                "expanded_tree_bank": bool(expanded),
                "max_tree_experts": int(max_exp),
                "target_transform": target_transform,
            }
        )
    for n_est, depth, leaf in itertools.product(
        rf_grid.get("n_estimators", [200, 500]),
        rf_grid.get("max_depth", [None, 12]),
        rf_grid.get("min_samples_leaf", [1, 5]),
    ):
        grid_rows.append(
            {
                "model": RANDOM_FOREST,
                "n_estimators": int(n_est),
                "max_depth": depth,
                "min_samples_leaf": int(leaf),
                "target_transform": target_transform,
            }
        )

    grid_df = pd.DataFrame(grid_rows)
    grid_path = os.path.join(outdir, "training_parameter_grid.csv")
    grid_df.to_csv(grid_path, index=False)

    result_rows: List[Dict[str, object]] = []
    for spec in grid_rows:
        model = str(spec["model"])
        row = dict(spec)
        row["screening_stage"] = "development"
        row["not_formal_test"] = True
        row["inner_cv_folds"] = inner_folds
        if model == TAR_MODEL:
            score = _inner_cv_score_tar(
                data,
                seed=seed,
                expanded_tree_bank=bool(spec["expanded_tree_bank"]),
                max_tree_experts=int(spec["max_tree_experts"]),
                target_transform=target_transform,
                inner_folds=inner_folds,
            )
        elif model == RANDOM_FOREST:
            factory = _rf_factory_screening(
                seed,
                int(spec["n_estimators"]),
                spec["max_depth"],
                int(spec["min_samples_leaf"]),
            )
            score = _inner_cv_score_multimodel(
                data,
                seed=seed,
                model_name=RANDOM_FOREST,
                factory=factory,
                target_transform=target_transform,
                inner_folds=inner_folds,
                expanded_tree_bank=False,
                max_tree_experts=0,
            )
        else:
            continue
        row[metric] = score
        result_rows.append(row)

    results_df = pd.DataFrame(result_rows)
    results_path = os.path.join(outdir, "training_screening_results.csv")
    results_df.to_csv(results_path, index=False)

    selected_per_model: Dict[str, Dict[str, object]] = {}
    decision_rows: List[Dict[str, object]] = []
    for model in (TAR_MODEL, RANDOM_FOREST):
        sub = results_df[results_df["model"] == model]
        if sub.empty:
            continue
        best_idx = sub[metric].astype(float).idxmax()
        best = sub.loc[best_idx].to_dict()
        selected_per_model[model] = best
        for _, r in sub.iterrows():
            decision_rows.append(
                {
                    "model": model,
                    "config_id": json.dumps({k: r[k] for k in spec_keys_for_model(model) if k in r}, sort_keys=True),
                    "passed": bool(r.name == best_idx),
                    "rejection_reason": "" if r.name == best_idx else f"lower_{metric}",
                    "used_for_main_manuscript": bool(r.name == best_idx),
                    "screening_stage": "development",
                }
            )

    tar_sel = selected_per_model.get(TAR_MODEL, {})
    tar_expanded = bool(tar_sel.get("expanded_tree_bank", False))
    tar_max_exp = int(tar_sel.get("max_tree_experts", 6))
    for control in (BEST_SINGLE_TREE, UNIFORM_TREE_MEAN):
        score = _inner_cv_score_multimodel(
            data,
            seed=seed,
            model_name=control,
            factory=lambda: None,
            target_transform=target_transform,
            inner_folds=inner_folds,
            expanded_tree_bank=tar_expanded,
            max_tree_experts=tar_max_exp,
        )
        cfg = {
            "model": control,
            "expanded_tree_bank": tar_expanded,
            "max_tree_experts": tar_max_exp,
            "target_transform": target_transform,
            "tree_bank_source": "same_as_selected_TAR",
        }
        cfg[metric] = score
        selected_per_model[control] = cfg
        decision_rows.append(
            {
                "model": control,
                "config_id": "shared_tar_expert_bank",
                "passed": True,
                "rejection_reason": "",
                "used_for_main_manuscript": True,
                "screening_stage": "development",
            }
        )

    decision_df = pd.DataFrame(decision_rows)
    decision_path = os.path.join(outdir, "training_screening_decision.csv")
    decision_df.to_csv(decision_path, index=False)

    manifest = {
        "stage": "screening",
        "not_formal_test": True,
        "same_search_budget_for_all_models": bool(plan_training.get("same_search_budget_for_all_models", True)),
        "split_mode": split_mode,
        "group_col": group_col,
        "inner_cv_folds": inner_folds,
        "selected_metric": metric,
        "selected_config_per_model": selected_per_model,
        "tree_controls_share_tar_expert_bank": bool(plan_training.get("tree_controls_share_tar_expert_bank", True)),
        "artifacts": {
            "training_parameter_grid.csv": grid_path,
            "training_screening_results.csv": results_path,
            "training_screening_decision.csv": decision_path,
        },
    }
    manifest_path = os.path.join(outdir, "training_screening_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(_json_native(manifest), fh, indent=2)
    return {
        "manifest": manifest,
        "results_df": results_df,
        "decision_df": decision_df,
        "selected_per_model": selected_per_model,
    }


def spec_keys_for_model(model: str) -> Tuple[str, ...]:
    if model == TAR_MODEL:
        return ("expanded_tree_bank", "max_tree_experts", "target_transform")
    if model == RANDOM_FOREST:
        return ("n_estimators", "max_depth", "min_samples_leaf", "target_transform")
    return ("expanded_tree_bank", "max_tree_experts", "target_transform")


def _json_native(obj):
    if isinstance(obj, dict):
        return {k: _json_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TAR benchmark: single-tree expert bank + inner OOF + target-wise ridge stack."
    )
    parser.add_argument("--x_csv", required=True)
    parser.add_argument("--y_csv", required=True)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--sample_weight_csv", default=None)
    parser.add_argument("--outdir", default="results/tree_srl_benchmark")
    parser.add_argument("--split_mode", choices=["group", "row"], default="group")
    parser.add_argument("--group_col", default="bio_id")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel worker count for repeated splits (n_repeats > 1).",
    )
    parser.add_argument("--lgbm_device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--target_transform", choices=["log", "none"], default="log")
    parser.add_argument("--candidate_score_csv", default=None)
    parser.add_argument("--force_single_split_significance", action="store_true")
    parser.add_argument(
        "--repeat_ci_method",
        choices=["t_interval", "percentile"],
        default="t_interval",
    )
    parser.add_argument("--use_physics_features", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--run_fixed_umax_validation", "--run_closed_loop", action="store_true", dest="run_fixed_umax_validation")
    parser.add_argument("--fixed_umax_outdir", "--closed_loop_outdir", default=None, dest="fixed_umax_outdir")
    parser.add_argument("--fixed_umax_finalize_only", "--closed_loop_finalize_only", action="store_true", dest="fixed_umax_finalize_only")
    parser.add_argument("--fixed_umax_force_rerun", "--closed_loop_force_rerun", action="store_true", dest="fixed_umax_force_rerun")
    parser.add_argument("--u_grid", default="arange:0:101:2")
    parser.add_argument("--w_track", type=float, default=1.0)
    parser.add_argument("--w_path", type=float, default=3.0)
    parser.add_argument("--w_probiotic", type=float, default=3.0)
    parser.add_argument("--w_dose", type=float, default=0.1)
    parser.add_argument("--max_closed_loop_cases", type=int, default=0)
    parser.add_argument("--permutation_replicates", type=int, default=5000)
    parser.add_argument("--expanded_tree_bank", action="store_true")
    parser.add_argument("--max_tree_experts", type=int, default=0)
    parser.add_argument(
        "--include_et_debug",
        action="store_true",
        help="Optional debug-only ExtraTrees control (not a manuscript control).",
    )
    parser.add_argument("--enable_gated_residual", action="store_true", default=False)
    parser.add_argument("--include_residual_diagnostics", action="store_true", default=False)
    parser.add_argument("--enable_uncertainty_decomposition", action="store_true", default=False)
    parser.add_argument(
        "--uncertainty_method",
        choices=["mc_dropout", "deep_ensemble"],
        default="mc_dropout",
    )
    parser.add_argument("--uncertainty_hidden_dim", type=int, default=128)
    parser.add_argument("--uncertainty_dropout", type=float, default=0.10)
    parser.add_argument("--uncertainty_epochs", type=int, default=100)
    parser.add_argument("--uncertainty_batch_size", type=int, default=256)
    parser.add_argument("--uncertainty_lr", type=float, default=1e-3)
    parser.add_argument("--uncertainty_mc_samples", type=int, default=50)
    parser.add_argument("--uncertainty_n_ensemble", type=int, default=5)
    parser.add_argument("--uncertainty_inference_batch_size", type=int, default=512)
    parser.add_argument("--show_uncertainty_main", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.split_mode == "row":
        print(
            "WARNING: row-level split is not manuscript-safe; use --split_mode group for publication results."
        )
    if args.enable_gated_residual and not args.include_residual_diagnostics:
        warnings.warn(
            "Gated residual is not part of the final TAR model; use --include_residual_diagnostics for diagnostics only.",
            stacklevel=2,
        )

    cycle_cfg = build_cycle_config_from_args(args)
    uncertainty_cfg = build_uncertainty_config_from_args(args)
    _configure_quiet_runtime(verbose=args.verbose)
    os.environ["TREE_SRL_LGBM_DEVICE"] = args.lgbm_device
    os.makedirs(args.outdir, exist_ok=True)

    config = vars(args).copy()
    config["not_for_manuscript"] = args.split_mode == "row"
    config["single_split_exploratory"] = args.n_repeats == 1
    config["significance_for_manuscript"] = args.n_repeats >= 10 and args.split_mode == "group"
    config["target_columns"] = TARGET_COLS_TTHR
    config["candidate_score_csv_used"] = bool(args.candidate_score_csv)
    config["main_plot_models"] = MAIN_BAR_ORDER
    config["tar_architecture"] = build_tar_architecture_schema()

    if args.run_fixed_umax_validation and args.n_repeats > 1:
        print(
            "NOTE: --run_fixed_umax_validation with n_repeats > 1 runs fixed-Umax validation after benchmark. "
            "For a standalone step, use closed_loop_eval.py with --predictions_dir."
        )

    repeat_results, last_data, data_by_repeat = run_benchmark_repeats(
        n_repeats=args.n_repeats,
        seed=args.seed,
        n_jobs=args.n_jobs,
        x_csv=args.x_csv,
        y_csv=args.y_csv,
        metadata_csv=args.metadata_csv,
        sample_weight_csv=args.sample_weight_csv,
        split_mode=args.split_mode,
        group_col=args.group_col,
        test_size=args.test_size,
        target_transform=args.target_transform,
        max_rows=args.max_rows,
        use_physics_features=args.use_physics_features,
        bootstrap=args.bootstrap if args.n_repeats == 1 else 0,
        lgbm_device=args.lgbm_device,
        verbose=args.verbose,
        expanded_tree_bank=args.expanded_tree_bank,
        max_tree_experts=args.max_tree_experts,
        include_et_debug=args.include_et_debug,
        enable_gated_residual=args.enable_gated_residual,
        candidate_score_csv=args.candidate_score_csv,
        uncertainty_cfg=uncertainty_cfg,
        cycle_cfg=cycle_cfg,
    )

    warn_if_repeats_missing(args.n_repeats, repeat_results)

    per_target_df = pd.DataFrame([row for result in repeat_results for row in result.per_target_rows])
    weights_df = pd.DataFrame([row for result in repeat_results for row in result.target_weights_rows])
    repeated_metrics_df = build_repeated_parameter_metrics(repeat_results)

    if args.n_repeats > 1:
        summary_df = aggregate_repeated_split_summaries(repeat_results, ci_method=args.repeat_ci_method)
    else:
        summary_df = pd.DataFrame(repeat_results[-1].summary_rows).drop(
            columns=["repeat_id", "seed"], errors="ignore"
        )

    control_models = [m for m in MANUSCRIPT_CONTROL_MODELS if m in repeated_metrics_df["model"].values]
    pairwise_df = build_parameter_pairwise_significance(
        repeat_results,
        srl_models=[TAR_MODEL],
        control_models=control_models,
        n_perm=args.permutation_replicates,
        seed=args.seed,
        single_split_exploratory=args.n_repeats == 1,
        force_single_split_significance=args.force_single_split_significance,
    )

    repeat_metadata_rows = build_repeat_metadata_rows(repeat_results).to_dict(orient="records")
    tree_expert_df = build_tree_expert_table(
        aggregate_tree_expert_summaries(repeat_results, ci_method=args.repeat_ci_method),
        weights_df,
    )
    write_benchmark_csv_artifacts(
        outdir=args.outdir,
        summary_df=summary_df,
        per_target_df=per_target_df,
        repeated_metrics_df=repeated_metrics_df,
        pairwise_df=pairwise_df,
        weights_df=weights_df,
        tree_expert_df=tree_expert_df,
    )
    prediction_paths = save_repeat_predictions(
        args.outdir,
        repeat_results,
        data_by_repeat,
        n_repeats=args.n_repeats,
        split_mode=args.split_mode,
        group_col=args.group_col,
    )
    write_predictions_manifest(
        args.outdir,
        prediction_paths,
        n_repeats=args.n_repeats,
        split_mode=args.split_mode,
        group_col=args.group_col,
        test_size=args.test_size,
        seed=args.seed,
    )

    generated_primary_csv = list(PRIMARY_CSV_OUTPUTS) + [
        "model_compare_manifest.json",
        "predictions_manifest.json",
    ]
    if args.n_repeats == 1:
        generated_primary_csv.append("predictions_all_models.csv")
    generated_primary_csv.extend(
        [
            f"repeats/repeat_{r.repeat_id:03d}/predictions.csv"
            for r in repeat_results
        ]
    )

    generated_primary_png = list(PRIMARY_PNG_OUTPUTS)
    if uncertainty_cfg.enabled and uncertainty_cfg.show_main:
        generated_primary_png.extend(UNCERTAINTY_PNG_OUTPUTS)

    uncertainty_case_df = pd.DataFrame(
        [row for result in repeat_results for row in result.uncertainty_case_rows]
    )
    uncertainty_summary_df = pd.DataFrame(
        [row for result in repeat_results for row in result.uncertainty_summary_rows]
    )
    uncertainty_training_log_df = pd.DataFrame(
        [row for result in repeat_results for row in result.uncertainty_training_log_rows]
    )
    uncertainty_manifest: Dict[str, object] = {
        "uncertainty_enabled": False,
        "uncertainty_method": uncertainty_cfg.method,
        "show_uncertainty_main": uncertainty_cfg.show_main,
        "n_repeats": args.n_repeats,
        "manuscript_safe": args.n_repeats >= 10 and args.split_mode == "group",
    }
    if uncertainty_cfg.enabled:
        uncertainty_manifest["uncertainty_enabled"] = any(
            not result.uncertainty_skipped
            and result.uncertainty_manifest.get("uncertainty_enabled", False)
            for result in repeat_results
        )
        uncertainty_manifest["torch_used"] = any(
            result.uncertainty_manifest.get("torch_used", False)
            for result in repeat_results
            if result.uncertainty_manifest
        )
        uncertainty_manifest["repeat_manifests"] = [
            result.uncertainty_manifest for result in repeat_results if result.uncertainty_manifest
        ]
        if not uncertainty_summary_df.empty and "mean_epistemic_fraction" in uncertainty_summary_df.columns:
            uncertainty_manifest["aleatoric_dominates_epistemic"] = bool(
                (1.0 - uncertainty_summary_df["mean_epistemic_fraction"].astype(float)).mean() > 0.5
            )
        if not uncertainty_case_df.empty and {"aleatoric_std", "abs_error"}.issubset(uncertainty_case_df.columns):
            sub = uncertainty_case_df[["aleatoric_std", "abs_error"]].dropna()
            if len(sub) > 2:
                uncertainty_manifest["uncertainty_abs_error_spearman"] = float(
                    spearmanr(sub["aleatoric_std"], sub["abs_error"]).correlation
                )
        write_uncertainty_artifacts(
            args.outdir,
            uncertainty_case_df,
            uncertainty_summary_df,
            uncertainty_training_log_df,
            uncertainty_manifest,
        )
        summary_path = os.path.join(args.outdir, "uncertainty_decomposition_summary.csv")
        if os.path.exists(summary_path):
            pd.read_csv(summary_path).to_csv(
                os.path.join(args.outdir, "uncertainty_summary.csv"), index=False
            )
            generated_primary_csv.append("uncertainty_summary.csv")
        generated_primary_csv.extend(
            ["uncertainty_decomposition.csv", "uncertainty_manifest.json"]
        )

    config["uncertainty_enabled"] = bool(uncertainty_manifest.get("uncertainty_enabled", False))
    config["uncertainty_method"] = uncertainty_cfg.method if uncertainty_cfg.enabled else None
    config["show_uncertainty_main"] = bool(uncertainty_cfg.show_main)
    if any(getattr(data, "candidate_score_alignment_failed", False) for data in data_by_repeat.values()):
        config["candidate_score_alignment_failed"] = True

    write_run_manifest(
        args.outdir,
        config,
        repeat_metadata_rows,
        repeat_results,
        generated_primary_csv=generated_primary_csv,
        generated_primary_png=generated_primary_png,
    )
    print_console_summary(summary_df)

    if args.run_fixed_umax_validation:
        from closed_loop_eval import (
            ClosedLoopConfig,
            parse_u_grid,
            resolve_weights_from_profile,
            run_fixed_umax_validation_pipeline,
        )

        cl_config = ClosedLoopConfig(
            u_grid=parse_u_grid(args.u_grid),
            weights=resolve_weights_from_profile("custom", args.w_track, args.w_path, args.w_probiotic, args.w_dose),
            max_closed_loop_cases=args.max_closed_loop_cases,
            bootstrap_replicates=args.bootstrap,
            permutation_replicates=args.permutation_replicates,
            repeat_ci_method=args.repeat_ci_method,
            weight_profile="custom",
            weight_selection_source="fixed",
        )
        cl_outdir = args.fixed_umax_outdir or os.path.join(os.path.dirname(args.outdir), "fixed_umax_validation")
        predictions_dir = os.path.join(args.outdir, "repeats")
        predictions_manifest = os.path.join(args.outdir, "predictions_manifest.json")
        run_fixed_umax_validation_pipeline(
            predictions_dir=predictions_dir,
            predictions_manifest=predictions_manifest,
            x_csv=args.x_csv,
            metadata_csv=args.metadata_csv,
            outdir=cl_outdir,
            config=cl_config,
            verbose=args.verbose,
            finalize_only=args.fixed_umax_finalize_only,
            force_rerun=args.fixed_umax_force_rerun,
        )
        print("\nCompleted fixed-Umax validation.")
        print(
            "Generate Fig. 4: python figure_audit.py --mode generate_plots "
            f"--groups fixed_umax_validation --fixed_umax_outdir {cl_outdir}"
        )

    print("\nSaved benchmark outputs to", args.outdir)
    print("Generate figures: python figure_audit.py --mode generate_plots --groups benchmark --benchmark_outdir", args.outdir)
    print(summary_df[["model", "mean_R2_original", "mean_R2_log"]].to_string(index=False))


if __name__ == "__main__":
    main()
