from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def ts_train_test_split(
    df: pd.DataFrame, test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into train and test sets based on a ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame sorted chronologically.
    test_ratio : float, optional
        Fraction of rows to allocate to the test set.
        Must be in the range (0.0, 1.0]. Default is 0.2 (20%).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df), where the test set contains the last `test_ratio` fraction
        of rows from the DataFrame.

    Raises
    ------
    ValueError
        If test_ratio is not within (0, 1).
    """
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1 (exclusive).")

    test_size = max(1, int(len(df) * test_ratio))

    return df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()


def _cat_indices(df: pd.DataFrame, cols: list[str]) -> list[int]:
    """Return indices (0..len(cols)-1) of categorical columns among `cols`."""
    cats = []
    for i, c in enumerate(cols):
        dt = df[c].dtype
        if (
            pd.api.types.is_object_dtype(dt)
            or pd.api.types.is_categorical_dtype(dt)
            or pd.api.types.is_bool_dtype(dt)
        ):
            cats.append(i)
    return cats


def train_regression_time_to_roi_models(
    df: pd.DataFrame,
    selected_features: dict[str, list[str]],
    targets_prefix: str = "time_to_",
    date_col: str = "Date",
    test_ratio: float = 0.2,
) -> dict[str, dict[str, object]]:
    """
    Train CatBoost regressors for time-to-ROI targets with a chronological split.

    This function detects continuous targets whose names contain `targets_prefix`
    (e.g., "time_to_10pct_months") and trains one CatBoostRegressor per target,
    using only the target-specific selected features and a tail-based time split
    (no shuffle) to avoid look-ahead.

    For each target, it reports MAE, MAPE, and R² on the test tail and appends a
    `{target}_pred` column with predictions to the returned test DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature-enriched dataset containing time-to-ROI targets and predictors.
        Assumes features are already leakage-safe (shifted) and numeric/categorical
        as expected by CatBoost.
    selected_features : dict[str, list[str]]
        Mapping: target column -> list of feature names to use for that model.
        Only columns present in `df` are used; `date_col` is always excluded.
    targets_prefix : str, optional
        Substring used to identify continuous targets. Default is "time_to_".
    date_col : str, optional
        Name of the date column (excluded from features). Default "Date".
    test_ratio : float, optional
        Fraction of the most recent rows used as the test set. Default 0.2.

    Returns
    -------
    results : dict[str, dict[str, object]]
        Per-target dictionary with:
          - "model": fitted CatBoostRegressor
          - "features": features used
          - "categorical_idx": indices of categorical features passed to CatBoost
          - "metrics": {
                "MAE": float,
                "MAPE": float,   # sklearn.mean_absolute_percentage_error
                "R2": float
            }
    test_df : pandas.DataFrame
        Copy of the test split augmented with `{target}_pred` columns for each
        trained target.
    """
    # determine targets
    targets = [c for c in df.columns if targets_prefix in c]
    # Default CatBoost params
    default_params = dict(
        loss_function="RMSE",
        random_seed=42,
        od_type="Iter",
        od_wait=50,
        verbose=False,
        allow_writing_files=False,
    )

    # Chronological split once for all targets
    train_df, test_df = ts_train_test_split(df, test_ratio=test_ratio)

    results = {}
    for tgt in targets:
        logger.info("Training target: %s", tgt)
        feats = [
            c
            for c in selected_features.get(tgt, [])
            if c in df.columns and c != date_col
        ]
        if not feats:
            raise ValueError(
                f"No features found for target '{tgt}' in selected_features."
            )
        cat_idx = _cat_indices(df, feats)

        # Build CatBoost Pools
        train_pool = Pool(
            train_df[feats],
            label=train_df[tgt],
            cat_features=cat_idx if cat_idx else None,
        )
        test_pool = Pool(
            test_df[feats],
            label=test_df[tgt],
            cat_features=cat_idx if cat_idx else None,
        )

        model = CatBoostRegressor(**default_params)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        pred = model.predict(test_pool)
        metrics = {
            "MAE": float(mean_absolute_error(test_df[tgt], pred)),
            "R2": float(r2_score(test_df[tgt], pred)),
            "MAPE": float(mean_absolute_percentage_error(test_df[tgt], pred)),
        }

        results[tgt] = {
            "model": model,
            "features": feats,
            "categorical_idx": cat_idx,
            "metrics": metrics,
        }
        # add predictions to test set
        test_df[f"{tgt}_pred"] = pred
    logger.info(results)

    return results, test_df


def train_classification_roi_within_horizon(
    df: pd.DataFrame,
    selected_features: dict[str, list[str]],
    targets_prefix: str = "roi_",
    date_col: str = "Date",
    test_ratio: float = 0.2,
) -> dict[str, dict[str, object]]:
    """
    Train CatBoost binary classifiers for "ROI reached within horizon" targets.

    This function detects binary targets whose column names contain
    `targets_prefix` (e.g., "roi_10pct_within_12"), then trains a separate
    CatBoost model per target using only its selected features and a
    chronological train/test split.

    The function safely handles single-class edge cases:
    - If the training set contains only one class → model training is skipped,
      and a constant probability baseline (prevalence) is returned.
    - If the test set contains only one class → AUC and PR-AUC cannot be
      computed; they are set to NaN.

    Model evaluation is based on the test tail to avoid look-ahead bias.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with features and binary ROI targets.
    selected_features : dict[str, list[str]]
        Mapping from target column name → features to use for that model.
        Must include only columns present in `df`.
    targets_prefix : str, optional
        Prefix used to identify ROI classification targets.
        Defaults to "roi_".
    date_col : str, optional
        Name of the date column that is excluded from features.
        Defaults to "Date".
    test_ratio : float, optional
        Fraction of the **most recent** rows used as the test set.
        Defaults to 0.2.

    Returns
    -------
    results : dict[str, dict[str, object]]
        Dictionary keyed by target column, where each value contains:
            - "status" : "trained" or "skipped_single_class_train"
            - "reason" : None or explanation for skipping
            - "model" : fitted CatBoostClassifier or None
            - "features" : List of feature names used
            - "metrics" : {'AUC', 'PR_AUC'} with NaN when undefined
    test_df : pandas.DataFrame
        Copy of test split augmented with prediction columns
        `{target}_proba_pred` for each trained model.
    """
    # determine targets
    targets = [c for c in df.columns if targets_prefix in c]
    # Default CatBoost params
    default_params = dict(
        loss_function="Logloss",
        random_seed=42,
        od_type="Iter",
        od_wait=50,
        verbose=False,
        allow_writing_files=False,
    )

    train_df, test_df = ts_train_test_split(df, test_ratio=test_ratio)

    results = {}
    for tgt in targets:
        logger.info("Training target: %s", tgt)
        feats = [
            c
            for c in selected_features.get(tgt, [])
            if c in df.columns and c != date_col
        ]
        if not feats:
            raise ValueError(
                f"No features found for target '{tgt}' in selected_features."
            )
        cat_idx = _cat_indices(df, feats)

        y_tr = train_df[tgt].astype(int)
        y_te = test_df[tgt].astype(int)

        # If train has only one class -> skip model training and return baseline
        if y_tr.nunique() < 2:
            # Baseline probability = prevalence in train
            p1 = float(y_tr.mean())
            proba = np.full(len(y_te), p1, dtype=float)

            # Metrics (AUC/PR need both classes in test)
            auc = float("nan")
            pr_auc = float("nan")

            results[tgt] = {
                "status": "skipped_single_class_train",
                "reason": "Only one class present in training data; returned prevalence baseline.",
                "model": None,
                "features": feats,
                "metrics": {"AUC": auc, "PR_AUC": pr_auc},
            }
            continue

        train_pool = Pool(
            train_df[feats], label=y_tr, cat_features=cat_idx if cat_idx else None
        )
        test_pool = Pool(
            test_df[feats], label=y_te, cat_features=cat_idx if cat_idx else None
        )

        model = CatBoostClassifier(**default_params)
        model.fit(train_pool, eval_set=test_pool, use_best_model=True)

        proba = model.predict_proba(test_pool)[:, 1]

        if y_te.nunique() == 2:
            auc = float(roc_auc_score(y_te, proba))
            pr_auc = float(average_precision_score(y_te, proba))
        else:
            auc = float("nan")
            pr_auc = float("nan")

        results[tgt] = {
            "status": "trained",
            "reason": None,
            "model": model,
            "features": feats,
            "metrics": {"AUC": auc, "PR_AUC": pr_auc},
        }
        # add predictions to test set
        test_df[f"{tgt}_proba_pred"] = proba

    logger.info(results)

    return results, test_df
