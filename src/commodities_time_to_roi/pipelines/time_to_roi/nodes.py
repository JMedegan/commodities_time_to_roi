import pandas as pd
import logging
import numpy as np
logger = logging.getLogger(__name__)


def filter_last_300_months(
    df: pd.DataFrame,
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Filter a monthly gold price dataset to retain only the most recent 300 months.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame that must contain at least a date column and
        corresponding price data.
    date_col : str, optional (default="Date")
        The column name containing the monthly date values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only the most recent 300 months of data,
        sorted in ascending date order.

    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    filtered = df.tail(300)

    logger.info(
        f"Filtered (monthly) range: {filtered[date_col].min().date()} → {filtered[date_col].max().date()}"
    )

    return filtered.reset_index(drop=True)

def add_multi_roi_time_targets(
    df: pd.DataFrame,
    project_params: dict[any],
    price_col: str = "Price",
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Add multiple ROI-based regression targets:
    number of months until each ROI threshold is reached.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly gold price dataset sorted in ascending date order.
    project_params : Dict[str, Any]
        Dictionary containing:
        - "roi_targets" : List[float]
            ROI thresholds expressed as decimals (e.g., 0.10 for +10%).
    price_col : str, optional
        Column name containing price data.
    date_col : str, optional
        Date column name used to enforce sorting.

    Returns
    -------
    pandas.DataFrame
        DataFrame including target columns:
        time_to_{roi*100}pct_months
        NaN when ROI not reached (censored data).
    """
    df = df.copy()
    df = df.sort_values(date_col)
    prices = df[price_col].values
    n = len(df)

    for roi in project_params["roi_targets"]:
        col = f"time_to_{int(roi*100)}pct_months"
        df[col] = np.nan

        for i in range(n):
            current_price = prices[i]
            future_roi = (prices[i:] - current_price) / current_price

            idx = np.where(future_roi >= roi)[0]
            if len(idx) > 0:
                df.loc[i, col] = idx[0]

    return df

def add_multi_roi_classification_targets(
    df: pd.DataFrame,
    project_params: dict[any],
    horizon_months: int = 12,
    price_col: str = "Price",
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Add multiple binary ROI targets indicating whether each ROI
    threshold is reached within a fixed horizon.

    Assumes the data is sorted in ascending chronological order.
    A safety sort is applied to ensure correctness.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly gold price dataset.
    project_params : Dict[str, Any]
        Dictionary containing:
        - "roi_targets" : List[float]
            ROI thresholds expressed as decimals (e.g., 0.10 for +10%).
    horizon_months : int, optional
        Months ahead to evaluate ROI achievement.
    price_col : str, optional
        Price column name.
    date_col : str, optional
        Date column name used to enforce sorting.

    Returns
    -------
    pandas.DataFrame
        Adds target columns:
        roi_{roi*100}pct_within_{horizon_months}
    """
    df = df.copy()
    df = df.sort_values(date_col)

    prices = df[price_col].values
    n = len(df)

    for roi in project_params["roi_targets"]:
        col = f"roi_{int(roi*100)}pct_within_{horizon_months}"
        df[col] = False

        for i in range(n - horizon_months):
            current_price = prices[i]
            future_price = prices[i + horizon_months]
            roi_val = (future_price - current_price) / current_price
            df.loc[i, col] = roi_val >= roi

    return df


def add_censoring_flags(
    df: pd.DataFrame,
    project_params: dict[any],
) -> pd.DataFrame:
    """
    Add binary censoring flags for time-to-ROI regression targets.

    For each ROI threshold (e.g. 0.10 for +10%), a corresponding binary event
    column is created:
    - 1 indicates that the ROI was reached (not censored)
    - 0 indicates that the event did not occur (censored observation)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time-to-ROI target columns, sorted chronologically.
    project_params : Dict[str, Any]
        Dictionary containing:
        - "roi_targets" : List[float]
            ROI thresholds expressed as decimals (e.g., 0.10 for +10%).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame including new censoring flag columns:
        * event_{roi*100}pct

    """
    df = df.copy()

    for roi in project_params["roi_targets"]:
        col = f"time_to_{int(roi*100)}pct_months"
        flag_col = f"event_{int(roi*100)}pct"
        df[flag_col] = df[col].notna().astype(int)

    return df

def impute_censored_max(
    df: pd.DataFrame,
    project_params: dict[any],
) -> pd.DataFrame:
    """
    Impute censored time-to-ROI targets using a worst-case assumption.

    For each ROI threshold, missing values (NaN) in the corresponding
    time-to-ROI target column are treated as right-censored observations,
    meaning the ROI was not reached within the observed time window.

    This function replaces NaN values with:
    (maximum observed time among uncensored samples) + 1

    This preserves the ordering relationship where censored cases are
    interpreted as having longer waiting times than any observed successful
    sample, without artificially reducing perceived risk.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time-to-ROI target columns.
    project_params : Dict[str, Any]
        Dictionary containing:
        - "roi_targets" : List[float]
            ROI thresholds expressed as decimals (e.g., 0.10 for +10%).

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with censored values imputed for regression tasks.
    """
    df = df.copy()

    for roi in project_params["roi_targets"]:
        col = f"time_to_{int(roi*100)}pct_months"
        max_val = df[col].max()
        df[col] = df[col].fillna(max_val + 1)

    return df

def add_log_price(df: pd.DataFrame, price_column: str = "Price", log_column: str = "log_price") -> pd.DataFrame:
    """
    Add a log-transformed version of a price column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that must contain a price column.
    price_column : str, optional
        Name of the price column to transform. Default is "Price".
    log_column : str, optional
        Name of the output column to store the log-transformed values.
        Default is "log_price".

    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with an additional column containing
        the natural logarithm of the price values. Non-positive values are
        replaced with NaN.

    Raises
    ------
    KeyError
        If the specified price_column does not exist in the DataFrame.
    """
    if price_column not in df.columns:
        raise KeyError(f"Column '{price_column}' does not exist in the DataFrame.")

    df = df.copy()
    df[log_column] = np.where(df[price_column] > 0, np.log(df[price_column]), np.nan)
    return df
