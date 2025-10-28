from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def _disc_series_quantiles(
    s: pd.Series, n_bins: int = 10, min_unique: int = 10
) -> pd.Series:
    """
    Discretize a 1D series into quantile bins (returns integer codes).
    - Non-numeric or too-few-unique → categorical codes.
    - Handles constant series.
    """
    s = s.copy()
    # If all constant -> assign 0
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(s), 0, dtype="int32"), index=s.index)
    # Low unique -> categorical codes
    if s.nunique(dropna=True) <= min_unique:
        return s.astype("category").cat.codes.astype("int32")
    # Numeric with enough unique values -> quantile bins (deciles default)
    try:
        bins = pd.qcut(s, q=min(n_bins, s.nunique()), duplicates="drop")
        return bins.cat.codes.astype("int32")
    except Exception:
        # fallback to categorical encoding
        return s.astype("category").cat.codes.astype("int32")


def _nmi_on_codes(a: pd.Series, b: pd.Series) -> float:
    """
    NMI on already-discretized integer codes (returns 0 if invalid).
    """
    a = pd.Series(a).astype("int32")
    b = pd.Series(b).astype("int32")
    if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
        return 0.0
    mask = ~(a.isna() | b.isna())
    if mask.sum() == 0:
        return 0.0
    return float(
        normalized_mutual_info_score(a[mask], b[mask], average_method="arithmetic")
    )


def _nmi_on_codes(a: pd.Series, b: pd.Series) -> float:
    """NMI on integer-coded series; returns 0 if invalid."""
    a = pd.Series(a).astype("int32")
    b = pd.Series(b).astype("int32")
    if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
        return 0.0
    mask = ~(a.isna() | b.isna())
    if mask.sum() == 0:
        return 0.0
    return float(
        normalized_mutual_info_score(a[mask], b[mask], average_method="arithmetic")
    )


def _disc_series_quantiles(
    s: pd.Series, n_bins: int = 10, min_unique: int = 10
) -> pd.Series:
    """
    Robust quantile discretizer returning int codes.
    - Non-numeric or low-cardinality -> categorical codes
    - All-NaN or constant -> all zeros
    - Numeric -> qcut with safe fallback
    """
    s = s.copy()
    if s.isna().all() or s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s), dtype="int32"), index=s.index)

    if not pd.api.types.is_numeric_dtype(s) or s.nunique(dropna=True) <= min_unique:
        return s.astype("category").cat.codes.astype("int32")

    try:
        q = min(n_bins, s.nunique(dropna=True))
        return pd.qcut(s, q=q, duplicates="drop").cat.codes.astype("int32")
    except Exception:
        return s.astype("category").cat.codes.astype("int32")


def _compute_feature_target_nmi(
    df: pd.DataFrame,
    targets: Optional[list[str]] = None,
    targets_prefixes: Iterable[str] = ("time_to_", "roi_"),
    exclude_cols: Iterable[str] = ("Date",),
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """
    Compute Normalized Mutual Information (NMI) between all features and targets,
    plus the feature–feature NMI matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing targets and feature candidates.
    targets : List[str], optional
        Explicit target column names. If None, targets are inferred by prefix.
    targets_prefixes : Iterable[str], optional
        Prefixes to auto-detect targets when `targets` is None.
    exclude_cols : Iterable[str], optional
        Columns to exclude from candidate features (e.g., "Date").
    n_bins : int, optional
        Number of quantile bins used for discretization (default: 10).

    Returns
    -------
    nmi_feature_target : pandas.DataFrame
        NMI scores (features x targets), values in [0, 1].
    nmi_feature_feature : pandas.DataFrame
        Symmetric NMI matrix (features x features), values in [0, 1].

    """
    # Determine targets
    if targets is None:
        targets = [
            c for c in df.columns if any(c.startswith(p) for p in targets_prefixes)
        ]
    if not targets:
        raise ValueError(
            "No target columns found. Provide `targets` or adjust `targets_prefixes`."
        )

    # Candidate features = all columns minus targets & excludes
    excludes = set(exclude_cols) | set(targets)
    candidate_features = [c for c in df.columns if c not in excludes]
    if not candidate_features:
        raise ValueError(
            "No candidate features found after excluding targets and exclude_cols."
        )

    # Discretize once (features and targets)
    disc_feats = {
        f: _disc_series_quantiles(df[f], n_bins=n_bins) for f in candidate_features
    }
    disc_tgts = {t: _disc_series_quantiles(df[t], n_bins=n_bins) for t in targets}
    disc_feats_df = pd.DataFrame(disc_feats)
    disc_tgts_df = pd.DataFrame(disc_tgts)

    # --- Feature–Target NMI table ---
    data = {}
    for f in candidate_features:
        row = {}
        a = disc_feats_df[f]
        for t in targets:
            row[t] = _nmi_on_codes(a, disc_tgts_df[t])
        data[f] = row
    nmi_feature_target = pd.DataFrame.from_dict(data, orient="index")[targets]

    # --- Feature–Feature NMI matrix (symmetric) ---
    n = len(candidate_features)
    nmi_feature_feature = pd.DataFrame(
        np.eye(n, dtype=float),
        index=candidate_features,
        columns=candidate_features,
    )
    for i, a in enumerate(candidate_features):
        sa = disc_feats_df[a]
        for j in range(i + 1, n):
            b = candidate_features[j]
            sb = disc_feats_df[b]
            val = _nmi_on_codes(sa, sb)
            nmi_feature_feature.iat[i, j] = val
            nmi_feature_feature.iat[j, i] = val

    nmi_feature_feature = nmi_feature_feature.fillna(0.0)
    np.fill_diagonal(nmi_feature_feature.values, 0.0)
    return nmi_feature_target, nmi_feature_feature


def prune_features_per_target(
    df: pd.DataFrame,
    redundancy_threshold: float = 0.9,
    max_features: Optional[int] = None,
) -> dict[str, list[str]]:
    """
    Strict redundancy pruning per target. Greedy forward selection:

    - Rank features by relevance to each target (from `nmi_feature_target`)
    - Select the next feature in the ranking if it is not redundant with any
      already selected features (NMI >= threshold in `nmi_feature_feature`)
    - Once a feature is selected, ALL features redundant with it are removed
      from consideration for the rest of the loop.
    - No replacements, no reordering.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing targets and feature candidates.
    redundancy_threshold : float, optional
        NMI threshold above which features are considered redundant (default 0.9).
    max_features : int, optional
        Optional upper limit on selected features.

    Returns
    -------
    Dict[str, List[str]]
        target -> list of selected feature names (ordered by selection).
    """
    nmi_feature_target, nmi_feature_feature = _compute_feature_target_nmi(df)
    common_feats = nmi_feature_target.index.intersection(nmi_feature_feature.index)
    nmi_ft = nmi_feature_target.loc[common_feats]
    nmi_ff = nmi_feature_feature.loc[common_feats, common_feats]

    selected_per_target = {}

    for tgt in nmi_ft.columns:
        ranking = nmi_ft[tgt].sort_values(ascending=False).index.tolist()
        selected = []

        for f in ranking:
            if max_features is not None and len(selected) >= max_features:
                break

            # Redundancy check
            if any(nmi_ff.loc[f, s] >= redundancy_threshold for s in selected):
                continue  # skip redundant feature

            selected.append(f)

        selected_per_target[tgt] = selected

    return selected_per_target
