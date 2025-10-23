"""
This is a boilerplate pipeline 'time_to_roi'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    add_log_price,
    impute_censored_max,
    add_censoring_flags,
    add_multi_roi_classification_targets,
    add_multi_roi_time_targets,
    filter_last_300_months,
)

from .feature_engineering_toolkit import build_features


def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                func=filter_last_300_months,
                inputs="gold_prices_data",
                outputs="gold_prices_data_filtered",
            ),
            Node(
                func=add_multi_roi_time_targets,
                inputs=["gold_prices_data_filtered", "params:project_params"],
                outputs="data_with_roi_time_targets",
            ),
            Node(
                func=add_multi_roi_classification_targets,
                inputs=["data_with_roi_time_targets", "params:project_params"],
                outputs="data_with_roi_classif_targets",
            ),
            Node(
                func=add_censoring_flags,
                inputs=["data_with_roi_classif_targets", "params:project_params"],
                outputs="data_with_censoring_flags",
            ),
            Node(
                func=impute_censored_max,
                inputs=["data_with_censoring_flags", "params:project_params"],
                outputs="data_with_targets_censored",
            ),
            Node(
                func=add_log_price,
                inputs="data_with_targets_censored",
                outputs="data_with_log_price",
            ),
            Node(
                func=build_features,
                inputs="data_with_log_price",
                outputs="data_ready_for_modeling",
            ),
        ]
    )
