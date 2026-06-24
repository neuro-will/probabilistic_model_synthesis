"""Utilities for fold-aware figure-5 across-condition analysis."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None


LABEL_MAP = {
    "omr_forward": "F",
    "omr_right": "R",
    "omr_left": "L",
    "omr_f_ns": "F",
    "omr_r_ns": "R",
    "omr_l_ns": "L",
    "F": "F",
    "R": "R",
    "L": "L",
}
ORDERED_LABELS = ["F", "R", "L"]
LABEL_RANK = {"F": 0, "R": 1, "L": 2}


def _pretty(lbl: str) -> str:
    return LABEL_MAP.get(lbl, str(lbl))


def _order_key(lbl: str) -> Tuple[int, str]:
    return LABEL_RANK.get(_pretty(lbl), 999), str(lbl)


def discover_layout(base_dir: str | Path) -> Dict:
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise ValueError(f"base_dir does not exist: {base_dir}")

    train_conds = sorted(
        [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("omr_")],
        key=_order_key,
    )
    if len(train_conds) == 0:
        raise ValueError(f"No train-condition folders found under {base_dir}.")

    fold_names = sorted(
        [d.name for d in (base_dir / train_conds[0]).iterdir() if d.is_dir() and d.name.startswith("fold_")],
        key=lambda s: int(s.split("_")[1]),
    )
    if len(fold_names) == 0:
        raise ValueError(
            "No fold directories found. Expected paths like <base_dir>/<condition>/fold_<k>/subj_<id>/..."
        )

    first_fold_dir = base_dir / train_conds[0] / fold_names[0]
    test_subjs = sorted(
        int(re.search(r".*_(\d+)", d.name)[1])
        for d in first_fold_dir.iterdir()
        if d.is_dir() and re.match(r".*subj", d.name) is not None
    )
    if len(test_subjs) == 0:
        raise ValueError(f"No subject folders found under {first_fold_dir}.")

    return {
        "base_dir": base_dir,
        "train_conds": train_conds,
        "fold_names": fold_names,
        "test_subjs": test_subjs,
    }


def load_results(
    base_dir: str | Path,
    train_conds: List[str],
    fold_names: List[str],
    test_subjs: List[int],
    fit_types: List[str] | None = None,
    pp_file: str = "pp_fit_results.pt",
) -> Dict:
    if fit_types is None:
        fit_types = ["multi_cond", "single_cond"]

    base_dir = Path(base_dir)
    rs = {}
    for cond in train_conds:
        rs[cond] = {}
        for fold in fold_names:
            rs[cond][fold] = {}
            for subj in test_subjs:
                rs[cond][fold][subj] = {}
                for fit_type in fit_types:
                    fit_type_file = base_dir / cond / fold / f"subj_{subj}" / fit_type / pp_file
                    load_kwargs = {"map_location": torch.device("cpu")}
                    try:
                        rs[cond][fold][subj][fit_type] = torch.load(
                            fit_type_file, weights_only=False, **load_kwargs
                        )
                    except TypeError:
                        rs[cond][fold][subj][fit_type] = torch.load(fit_type_file, **load_kwargs)
    return rs


def build_elbo_arrays(
    rs: Dict,
    train_conds: List[str],
    fold_names: List[str],
    test_subjs: List[int],
    mdl_type: str = "ip",
    test_periods: List[str] | None = None,
) -> Dict:
    if test_periods is None:
        test_periods = ["omr_forward", "omr_right", "omr_left"]

    n_fish = len(test_subjs)
    n_folds = len(fold_names)
    n_c = len(ORDERED_LABELS)
    sb = np.full((n_fish, n_folds, n_c, n_c), np.nan, dtype=float)
    db = np.full((n_fish, n_folds, n_c, n_c), np.nan, dtype=float)

    for fi, subj in enumerate(test_subjs):
        for fo, fold in enumerate(fold_names):
            for train_cond in train_conds:
                tr_lbl = _pretty(train_cond)
                tr_i = ORDERED_LABELS.index(tr_lbl)

                single_period_elbos = rs[train_cond][fold][subj]["single_cond"][mdl_type]["period_elbo_vls"][subj]
                multi_period_elbos = rs[train_cond][fold][subj]["multi_cond"][mdl_type]["period_elbo_vls"][subj]

                for test_period in test_periods:
                    te_lbl = _pretty(test_period)
                    te_i = ORDERED_LABELS.index(te_lbl)
                    single_rec = single_period_elbos.get(test_period)
                    multi_rec = multi_period_elbos.get(test_period)
                    if single_rec is None or multi_rec is None:
                        continue
                    sb[fi, fo, tr_i, te_i] = single_rec["vl"]["elbo"].item() / single_rec["n_smps"]
                    db[fi, fo, tr_i, te_i] = multi_rec["vl"]["elbo"].item() / multi_rec["n_smps"]

    delta = db - sb
    diag_mask = np.eye(n_c, dtype=bool)
    offdiag_mask = ~diag_mask

    fish_fold_diag = np.nanmean(delta[:, :, diag_mask], axis=2)
    fish_fold_offdiag = np.nanmean(delta[:, :, offdiag_mask], axis=2)
    fish_fold_all = np.nanmean(delta.reshape(n_fish, n_folds, -1), axis=2)

    return {
        "sb": sb,
        "db": db,
        "delta": delta,
        "diag_mask": diag_mask,
        "offdiag_mask": offdiag_mask,
        "fish_fold_diag": fish_fold_diag,
        "fish_fold_offdiag": fish_fold_offdiag,
        "fish_fold_all": fish_fold_all,
    }


def exact_sign_flip_test(deltas: np.ndarray, alternative: str = "two-sided") -> Tuple[float, float]:
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    n = deltas.size
    if n == 0:
        return np.nan, np.nan

    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
    null_stats = (signs * deltas).mean(axis=1)
    observed = float(np.mean(deltas))

    if alternative == "greater":
        p_value = float(np.mean(null_stats >= observed))
    elif alternative == "less":
        p_value = float(np.mean(null_stats <= observed))
    elif alternative == "two-sided":
        p_value = float(np.mean(np.abs(null_stats) >= abs(observed)))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    return observed, p_value


def exact_sign_count_test(deltas: np.ndarray, alternative: str = "two-sided") -> Tuple[int, int, float]:
    """Exact sign test based only on the number of positive non-zero differences."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    deltas = deltas[deltas != 0]
    n = deltas.size
    if n == 0:
        return 0, 0, np.nan

    n_positive = int(np.sum(deltas > 0))
    counts = np.arange(n + 1)
    probs = np.asarray([np.math.comb(n, k) for k in counts], dtype=float) / (2 ** n)

    if alternative == "greater":
        p_value = float(np.sum(probs[counts >= n_positive]))
    elif alternative == "less":
        p_value = float(np.sum(probs[counts <= n_positive]))
    elif alternative == "two-sided":
        observed_dev = abs(n_positive - n / 2)
        p_value = float(np.sum(probs[np.abs(counts - n / 2) >= observed_dev]))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    return n_positive, int(n), p_value


def holm_correction(pvals: List[float]) -> List[float]:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(np.nan_to_num(pvals, nan=1.0))
    adjusted = np.full(m, np.nan, dtype=float)
    running_max = 0.0
    for i, idx in enumerate(order):
        if np.isnan(pvals[idx]):
            adjusted[idx] = np.nan
            continue
        adj = (m - i) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def primary_fish_level_test(
    fish_fold_offdiag: np.ndarray,
    fish_ids: List[int],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Primary inference: collapse folds within each fish, then test across fish."""
    fish_fold_offdiag = np.asarray(fish_fold_offdiag, dtype=float)
    fish_means = np.nanmean(fish_fold_offdiag, axis=1)
    observed, p = exact_sign_flip_test(fish_means, alternative=alternative)
    row = {
        "effect": "offdiag_db_minus_sb",
        "alternative": alternative,
        "n_fish": int(np.sum(~np.isnan(fish_means))),
        "mean_delta_offdiag": observed,
        "p_exact_sign_flip": p,
    }
    for fi, fish in enumerate(fish_ids):
        row[f"fish_{fish}_mean_delta"] = fish_means[fi]
    return pd.DataFrame([row])


def pooled_fish_fold_test(
    fish_fold_offdiag: np.ndarray,
    fish_ids: List[int],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Pooled fold-level inference: exact sign-flip over every valid fish-fold value."""
    fish_fold_offdiag = np.asarray(fish_fold_offdiag, dtype=float)
    pooled_values = fish_fold_offdiag.reshape(-1)
    observed, p = exact_sign_flip_test(pooled_values, alternative=alternative)

    row = {
        "effect": "offdiag_db_minus_sb",
        "unit": "fish_fold",
        "alternative": alternative,
        "n_fish": int(fish_fold_offdiag.shape[0]),
        "n_fish_folds": int(np.sum(~np.isnan(pooled_values))),
        "mean_delta_offdiag": observed,
        "p_exact_sign_flip": p,
    }

    for fi, fish in enumerate(fish_ids):
        fish_values = fish_fold_offdiag[fi, :]
        row[f"fish_{fish}_n_folds"] = int(np.sum(~np.isnan(fish_values)))
        row[f"fish_{fish}_mean_delta"] = float(np.nanmean(fish_values))
    return pd.DataFrame([row])


def per_fish_pair_type_tests(
    delta: np.ndarray,
    diag_mask: np.ndarray,
    offdiag_mask: np.ndarray,
    fish_ids: List[int],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Per-fish exact sign-count tests across condition-cell values."""
    rows = []
    raw_pvals = []
    delta = np.asarray(delta, dtype=float)
    for pair_type, mask in [("diagonal", diag_mask), ("off_diagonal", offdiag_mask)]:
        for fi, fish in enumerate(fish_ids):
            values = delta[fi, :, mask].reshape(-1)
            n_positive, n_nonzero, p = exact_sign_count_test(values, alternative=alternative)
            raw_pvals.append(p)
            rows.append(
                {
                    "fish": fish,
                    "pair_type": pair_type,
                    "alternative": alternative,
                    "n_comparisons": int(np.sum(~np.isnan(values))),
                    "n_nonzero_comparisons": n_nonzero,
                    "n_positive": n_positive,
                    "positive_fraction": n_positive / n_nonzero if n_nonzero > 0 else np.nan,
                    "p_raw": p,
                }
            )

    p_holm = holm_correction(raw_pvals)
    for row, p_corr in zip(rows, p_holm):
        row["p_holm"] = p_corr
    return pd.DataFrame(rows)


def pooled_pair_type_tests(
    delta: np.ndarray,
    diag_mask: np.ndarray,
    offdiag_mask: np.ndarray,
    fish_ids: List[int],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Pooled exact sign tests across condition-cell values."""
    rows = []
    raw_pvals = []
    delta = np.asarray(delta, dtype=float)
    for pair_type, mask in [("diagonal", diag_mask), ("off_diagonal", offdiag_mask)]:
        pooled_values = delta[:, :, mask].reshape(-1)
        n_positive, n_nonzero, p = exact_sign_count_test(pooled_values, alternative=alternative)
        raw_pvals.append(p)
        row = {
            "pair_type": pair_type,
            "unit": "condition_cell",
            "alternative": alternative,
            "n_fish": int(delta.shape[0]),
            "n_comparisons": int(np.sum(~np.isnan(pooled_values))),
            "n_nonzero_comparisons": n_nonzero,
            "n_positive": n_positive,
            "positive_fraction": n_positive / n_nonzero if n_nonzero > 0 else np.nan,
            "p_raw": p,
        }
        for fi, fish in enumerate(fish_ids):
            fish_values = delta[fi, :, mask].reshape(-1)
            row[f"fish_{fish}_n_comparisons"] = int(np.sum(~np.isnan(fish_values)))
            row[f"fish_{fish}_n_positive"] = int(np.sum(fish_values[~np.isnan(fish_values)] > 0))
        rows.append(row)

    p_holm = holm_correction(raw_pvals)
    for row, p_corr in zip(rows, p_holm):
        row["p_holm"] = p_corr
    return pd.DataFrame(rows)


def fold_consistency_tests(
    fish_fold_offdiag: np.ndarray,
    fish_ids: List[int],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Per-fish exact sign-flip tests across folds."""
    rows = []
    raw_pvals = []
    for fi, fish in enumerate(fish_ids):
        observed, p = exact_sign_flip_test(fish_fold_offdiag[fi, :], alternative=alternative)
        raw_pvals.append(p)
        rows.append(
            {
                "fish": fish,
                "alternative": alternative,
                "n_folds": int(np.sum(~np.isnan(fish_fold_offdiag[fi, :]))),
                "mean_delta_offdiag": observed,
                "p_raw": p,
            }
        )

    p_holm = holm_correction(raw_pvals)
    for row, p_corr in zip(rows, p_holm):
        row["p_holm"] = p_corr
    return pd.DataFrame(rows)


def primary_within_fish_tests(
    fish_fold_offdiag: np.ndarray,
    fish_ids: List[int],
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Backward-compatible alias for the fold-level consistency diagnostic."""
    return fold_consistency_tests(
        fish_fold_offdiag=fish_fold_offdiag,
        fish_ids=fish_ids,
        alternative=alternative,
    )


def secondary_mixed_model(
    delta: np.ndarray,
    fish_ids: List[int],
    fold_names: List[str],
    alternative: str = "two-sided",
) -> Dict:
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    n_fish, n_folds, n_train, n_test = delta.shape
    recs = []
    for fi in range(n_fish):
        for fo in range(n_folds):
            for tr in range(n_train):
                for te in range(n_test):
                    vl = delta[fi, fo, tr, te]
                    if np.isnan(vl):
                        continue
                    pair_type = "diag" if tr == te else "offdiag"
                    recs.append(
                        {
                            "fish": f"fish_{fish_ids[fi]}",
                            "fold": fold_names[fo],
                            "fish_fold": f"fish_{fish_ids[fi]}::{fold_names[fo]}",
                            "pair": f"{ORDERED_LABELS[tr]}->{ORDERED_LABELS[te]}",
                            "pair_type": pair_type,
                            "delta": float(vl),
                        }
                    )
    df = pd.DataFrame(recs)
    if df.empty:
        return {"ok": False, "reason": "No valid delta values for mixed model.", "df": df}

    if smf is None:
        # Fallback: exact sign-flip on fish summaries, not fish-fold summaries.
        summary = (
            df.groupby(["fish", "pair_type"])["delta"]
            .mean()
            .reset_index()
            .pivot_table(index=["fish"], columns="pair_type", values="delta")
            .reset_index()
        )
        overall_delta = summary[["diag", "offdiag"]].mean(axis=1).to_numpy()
        offdiag_minus_diag = (summary["offdiag"] - summary["diag"]).to_numpy()
        obs_overall, p_overall = exact_sign_flip_test(overall_delta, alternative=alternative)
        obs_pair, p_pair = exact_sign_flip_test(offdiag_minus_diag, alternative=alternative)
        p_holm = holm_correction([p_overall, p_pair])
        return {
            "ok": True,
            "mode": "exact_sign_flip_fallback",
            "alternative": alternative,
            "df": df,
            "overall_effect": {"estimate": obs_overall, "p_raw": p_overall, "p_holm": p_holm[0]},
            "offdiag_vs_diag": {"estimate": obs_pair, "p_raw": p_pair, "p_holm": p_holm[1]},
        }

    try:
        md = smf.mixedlm(
            "delta ~ C(pair_type)",
            data=df,
            groups=df["fish"],
            vc_formula={"fish_fold": "0 + C(fish_fold)", "pair": "0 + C(pair)"},
            re_formula="1",
        )
        fit = md.fit(reml=False, method="lbfgs", disp=False)
    except Exception as exc:
        return {"ok": False, "reason": f"MixedLM failed: {exc}", "df": df}

    def _p_value_for_alternative(beta: float, p_two_sided: float) -> float:
        if np.isnan(p_two_sided):
            return np.nan
        if alternative == "two-sided":
            return p_two_sided
        if alternative == "greater":
            if beta >= 0:
                return p_two_sided / 2.0
            return 1.0 - (p_two_sided / 2.0)
        if beta <= 0:
            return p_two_sided / 2.0
        return 1.0 - (p_two_sided / 2.0)

    b0 = float(fit.params.get("Intercept", np.nan))
    p0 = _p_value_for_alternative(b0, float(fit.pvalues.get("Intercept", np.nan)))
    b1 = float(fit.params.get("C(pair_type)[T.offdiag]", np.nan))
    p1 = _p_value_for_alternative(b1, float(fit.pvalues.get("C(pair_type)[T.offdiag]", np.nan)))
    p_holm = holm_correction([p0, p1])

    return {
        "ok": True,
        "mode": "mixedlm",
        "alternative": alternative,
        "df": df,
        "fit": fit,
        "overall_effect": {"estimate": b0, "p_raw": p0, "p_holm": p_holm[0]},
        "offdiag_vs_diag": {"estimate": b1, "p_raw": p1, "p_holm": p_holm[1]},
    }


def make_fold_effect_plot(
    fish_fold_diag: np.ndarray,
    fish_fold_offdiag: np.ndarray,
    fish_ids: List[int],
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    fig, axs = plt.subplots(1, 2, figsize=(4.4, 2.0), sharey=True, constrained_layout=True)
    fish_markers = ["o", "^", "s", "D", "P", "X"]
    fish_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown", "tab:gray"]

    for ax_i, (ax, arr, title) in enumerate(
        [
            (axs[0], fish_fold_diag, "Diagonal"),
            (axs[1], fish_fold_offdiag, "Off-diagonal"),
        ]
    ):
        for fi, fish in enumerate(fish_ids):
            x_center = fi + 1
            y = arr[fi, :]
            valid = ~np.isnan(y)
            x = np.linspace(-0.10, 0.10, np.sum(valid)) + x_center
            ax.scatter(
                x,
                y[valid],
                s=18,
                marker=fish_markers[fi % len(fish_markers)],
                color=fish_colors[fi % len(fish_colors)],
                alpha=0.75,
            )
            if np.any(valid):
                mn = np.mean(y[valid])
                ax.plot([x_center - 0.14, x_center + 0.14], [mn, mn], color="k", lw=1.0)

        ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax.set_title(title, fontsize=7)
        ax.set_xticks(np.arange(1, len(fish_ids) + 1), labels=[f"Fish {i+1}" for i in range(len(fish_ids))])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax_i == 0:
            ax.set_ylabel("ΔELBO (DB - SB)")

    if save_path is not None:
        fig.savefig(save_path, format="svg", dpi=500, bbox_inches="tight", pad_inches=0.05, transparent=True)
    return fig, axs


def make_condition_pair_plot(
    delta: np.ndarray,
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    n_fish, n_folds, n_cond, _ = delta.shape
    pair_labels = []
    pair_means = []
    pair_sems = []
    for tr in range(n_cond):
        for te in range(n_cond):
            pair_labels.append(f"{ORDERED_LABELS[tr]}→{ORDERED_LABELS[te]}")
            vals = delta[:, :, tr, te].reshape(-1)
            vals = vals[~np.isnan(vals)]
            pair_means.append(np.mean(vals))
            pair_sems.append(np.std(vals, ddof=1) / np.sqrt(max(len(vals), 1)))

    x = np.arange(len(pair_labels))
    fig, ax = plt.subplots(figsize=(4.8, 2.0), constrained_layout=True)
    ax.errorbar(x, pair_means, yerr=pair_sems, fmt="o", color="k", ms=3, lw=0.9, capsize=2)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xticks(x, labels=pair_labels)
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylabel("Mean ΔELBO")
    ax.set_xlabel("Train→Test pair")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if save_path is not None:
        fig.savefig(save_path, format="svg", dpi=500, bbox_inches="tight", pad_inches=0.05, transparent=True)
    return fig, ax


def make_heatmaps(
    delta: np.ndarray,
    save_mean_path: str | Path | None = None,
    save_count_path: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray, plt.Figure, np.ndarray]:
    mean_delta = np.nanmean(delta, axis=(0, 1))
    positive_counts = np.sum(delta > 0, axis=(0, 1))
    n_total = np.sum(~np.isnan(delta), axis=(0, 1))

    fig1, ax1 = plt.subplots(figsize=(1.8, 1.7), constrained_layout=True)
    im1 = ax1.imshow(mean_delta, cmap="bwr", norm=mcolors.CenteredNorm())
    ax1.set_xticks(np.arange(3), labels=ORDERED_LABELS)
    ax1.set_yticks(np.arange(3), labels=ORDERED_LABELS)
    ax1.set_xlabel("Test condition")
    ax1.set_ylabel("Target fish \n train condition")
    ax1.set_title("Mean ΔELBO", fontsize=7)
    for r in range(3):
        for c in range(3):
            ax1.text(c, r, f"{mean_delta[r, c]:.0f}", ha="center", va="center", fontsize=5)
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig2, ax2 = plt.subplots(figsize=(1.8, 1.7), constrained_layout=True)
    max_count = int(np.nanmax(n_total))
    count_bounds = np.arange(-0.5, max_count + 1.5, 1)
    count_ticks = np.array([0, 5, 10, 15])
    count_cmap = plt.get_cmap("Greens", max_count + 1)
    count_norm = mcolors.BoundaryNorm(count_bounds, count_cmap.N)
    im2 = ax2.imshow(positive_counts, cmap=count_cmap, norm=count_norm)
    ax2.set_xticks(np.arange(3), labels=ORDERED_LABELS)
    ax2.set_yticks(np.arange(3), labels=ORDERED_LABELS)
    ax2.set_xlabel("Test condition")
    ax2.set_ylabel("Target fish \n train condition")
    ax2.set_title("Count(ΔELBO > 0 \n across folds and fish)", fontsize=7)
    for r in range(3):
        for c in range(3):
            ax2.text(c, r, f"{int(positive_counts[r, c])}/{int(n_total[r, c])}", ha="center", va="center", fontsize=5)
    cbar2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, boundaries=count_bounds)
    cbar2.set_ticks(count_ticks)
    cbar2.set_ticklabels([str(int(tick)) for tick in count_ticks])
    cbar2.minorticks_off()

    if save_mean_path is not None:
        fig1.savefig(save_mean_path, format="svg", dpi=500, bbox_inches="tight", pad_inches=0.05, transparent=True)
    if save_count_path is not None:
        fig2.savefig(save_count_path, format="svg", dpi=500, bbox_inches="tight", pad_inches=0.05, transparent=True)
    return fig1, ax1, fig2, ax2


def make_per_fish_improvement_heatmaps(
    delta: np.ndarray,
    fish_ids: List[int],
    save_dir: str | Path | None = None,
    save_prefix: str = "elbo_improvements_subj_",
) -> List[Tuple[plt.Figure, plt.Axes]]:
    """Create one ΔELBO heatmap per fish, averaging across folds."""
    if delta.ndim != 4:
        raise ValueError("Expected delta shape: (fish, fold, train, test).")

    fish_mean = np.nanmean(delta, axis=1)
    vmax = np.nanmax(np.abs(fish_mean))
    vmax = float(vmax if np.isfinite(vmax) and vmax > 0 else 1.0)

    outputs = []
    for fi, fish in enumerate(fish_ids):
        mat = fish_mean[fi]
        fig, ax = plt.subplots(figsize=(1.5, 1.5), constrained_layout=True)
        im = ax.imshow(mat, cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(3), labels=ORDERED_LABELS)
        ax.set_yticks(np.arange(3), labels=ORDERED_LABELS)
        ax.set_xlabel("Test condition")
        ax.set_ylabel("Target fish\ntraining condition")
        ax.set_title(f"Fish {fi+1}", fontsize=6)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center", fontsize=5, color="black")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if save_dir is not None:
            save_path = Path(save_dir) / f"{save_prefix}{fish}.svg"
            fig.savefig(save_path, format="svg", dpi=500, bbox_inches="tight", pad_inches=0.05, transparent=True)
        outputs.append((fig, ax))
    return outputs


def make_sb_db_paired_by_condition_split(
    sb: np.ndarray,
    db: np.ndarray,
    fish_ids: List[int],
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Recreate paired SB vs DB plot grouped into diagonal and off-diagonal pairs."""
    n_fish, n_folds, n_cond, _ = sb.shape
    fish_markers = ["o", "^", "s", "D", "P", "X"]
    sb_color = "firebrick"
    db_color = "navy"
    fish_offsets = np.linspace(-0.06, 0.06, n_fish)
    fold_offsets = np.linspace(-0.025, 0.025, n_folds)
    sb_shift = -0.14
    db_shift = 0.14

    pairs_diag = [(tr, te) for tr in range(n_cond) for te in range(n_cond) if tr == te]
    pairs_off = [(tr, te) for tr in range(n_cond) for te in range(n_cond) if tr != te]

    def _draw_group(ax: plt.Axes, pairs: List[Tuple[int, int]], title: str):
        pair_centers = np.arange(len(pairs), dtype=float)
        for pi, (tr, te) in enumerate(pairs):
            center = pair_centers[pi]
            for fi in range(n_fish):
                mk = fish_markers[fi % len(fish_markers)]
                for fo in range(n_folds):
                    y_sb = sb[fi, fo, tr, te]
                    y_db = db[fi, fo, tr, te]
                    if np.isnan(y_sb) or np.isnan(y_db):
                        continue
                    x_base = center + fish_offsets[fi] + fold_offsets[fo]
                    x_sb = x_base + sb_shift
                    x_db = x_base + db_shift
                    ax.plot([x_sb, x_db], [y_sb, y_db], color="0.75", linewidth=0.6, zorder=1)
                    ax.scatter([x_sb], [y_sb], s=16, marker=mk, color=sb_color, zorder=2)
                    ax.scatter([x_db], [y_db], s=16, marker=mk, color=db_color, zorder=2)

            vals_sb = sb[:, :, tr, te].reshape(-1)
            vals_db = db[:, :, tr, te].reshape(-1)
            vals_sb = vals_sb[~np.isnan(vals_sb)]
            vals_db = vals_db[~np.isnan(vals_db)]
            if vals_sb.size > 0 and vals_db.size > 0:
                ax.scatter([center + sb_shift], [np.mean(vals_sb)], s=24, marker="D", color=sb_color,
                           edgecolors="k", linewidths=0.3, zorder=3)
                ax.scatter([center + db_shift], [np.mean(vals_db)], s=24, marker="D", color=db_color,
                           edgecolors="k", linewidths=0.3, zorder=3)

        pair_labels = [f"{ORDERED_LABELS[tr]}→{ORDERED_LABELS[te]}" for tr, te in pairs]
        ax.set_xticks(pair_centers, labels=pair_labels)
        ax.tick_params(axis="x", labelrotation=35)
        ax.set_title(title, fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(3.5, 1.8),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [max(1, len(pairs_diag)), max(1, len(pairs_off))]},
    )
    _draw_group(axs[0], pairs_diag, "Within-modality")
    _draw_group(axs[1], pairs_off, "Across-modalities")
    axs[0].set_ylabel("Normalized ELBO")
    for ax in axs:
        ax.set_xlabel("Train→Test condition pair")

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=sb_color, label="SB", markersize=4),
        Line2D([0], [0], marker="o", linestyle="None", color=db_color, label="DB", markersize=4),
    ]
    fish_handles = [
        Line2D([0], [0], marker=fish_markers[i], linestyle="None", color="k", label=f"Fish {i+1}", markersize=4)
        for i in range(n_fish)
    ]
    leg1 = axs[0].legend(handles=model_handles, frameon=False, fontsize=5, loc="upper left")
    axs[0].add_artist(leg1)
    axs[1].legend(handles=fish_handles, frameon=False, fontsize=5, loc="upper right", ncol=min(n_fish, 3))

    if save_path is not None:
        fig.savefig(save_path, format="svg", dpi=500, bbox_inches="tight", pad_inches=0.05, transparent=True)
    return fig, axs


def make_sb_db_identity_scatter(
    sb: np.ndarray,
    db: np.ndarray,
    fish_ids: List[int],
    save_path: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Identity scatter of SB vs DB over fish, folds, and condition pairs."""
    n_fish, n_folds, n_cond, _ = sb.shape
    fish_markers = ["o", "^", "s", "D", "P", "X"]
    pair_colors = plt.cm.tab10(np.linspace(0, 1, n_cond * n_cond))

    sb_vals = []
    db_vals = []
    pair_inds = []
    fish_inds = []
    for fi in range(n_fish):
        for fo in range(n_folds):
            for tr in range(n_cond):
                for te in range(n_cond):
                    s = sb[fi, fo, tr, te]
                    d = db[fi, fo, tr, te]
                    if np.isnan(s) or np.isnan(d):
                        continue
                    sb_vals.append(s)
                    db_vals.append(d)
                    pair_inds.append(tr * n_cond + te)
                    fish_inds.append(fi)
    sb_vals = np.asarray(sb_vals)
    db_vals = np.asarray(db_vals)

    fig, ax = plt.subplots(figsize=(3.0, 2.6), constrained_layout=True)
    for x, y, pi, fi in zip(sb_vals, db_vals, pair_inds, fish_inds):
        ax.scatter(
            x,
            y,
            s=24,
            marker=fish_markers[fi % len(fish_markers)],
            color=pair_colors[pi],
            alpha=0.85,
            linewidths=0.0,
        )

    if sb_vals.size > 0:
        vmin = min(float(np.min(sb_vals)), float(np.min(db_vals)))
        vmax = max(float(np.max(sb_vals)), float(np.max(db_vals)))
    else:
        vmin, vmax = -1.0, 1.0
    pad = 0.03 * (vmax - vmin if vmax > vmin else 1.0)
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], "k--", linewidth=0.8, zorder=0)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_xlabel("SB performance (single_cond ELBO)")
    ax.set_ylabel("DB performance (multi_cond ELBO)")
    ax.set_title("DB vs SB across fish, folds and condition pairs", fontsize=6)
    ax.text(0.02, 0.98, "Above y=x: DB better", transform=ax.transAxes, va="top", fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fish_handles = [
        Line2D([0], [0], marker=fish_markers[i], linestyle="None", color="k", label=f"Fish {i+1}", markersize=4)
        for i in range(n_fish)
    ]
    leg1 = ax.legend(handles=fish_handles, frameon=False, fontsize=5, loc="upper left", ncol=min(n_fish, 3))
    ax.add_artist(leg1)

    if save_path is not None:
        fig.savefig(save_path, format="svg", dpi=500, bbox_inches="tight", pad_inches=0.05, transparent=True)
    return fig, ax
