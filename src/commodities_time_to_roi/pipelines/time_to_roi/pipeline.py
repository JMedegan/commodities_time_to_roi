"""
This is a boilerplate pipeline 'time_to_roi'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa

from .nodes import (
    impute_censored,
    add_multi_roi_classification_targets,
    add_multi_roi_time_targets,
    filter_last_300_months,
)

from .feature_engineering_toolkit import build_features
from .feature_selection import prune_features_per_target
from .model_training import (
    train_classification_roi_within_horizon,
    train_regression_time_to_roi_models,
)


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
                func=impute_censored,
                inputs=["data_with_roi_classif_targets", "params:project_params"],
                outputs="data_with_targets_censored",
            ),
            Node(
                func=build_features,
                inputs="data_with_targets_censored",
                outputs="data_ready_for_modeling",
            ),
            Node(
                func=prune_features_per_target,
                inputs="data_ready_for_modeling",
                outputs="selected_features_per_target",
            ),
            Node(
                func=train_regression_time_to_roi_models,
                inputs=["data_ready_for_modeling", "selected_features_per_target"],
                outputs=["time_to_roi_results", "test_set_with_time_to_roi_preds"],
            ),
            Node(
                func=train_classification_roi_within_horizon,
                inputs=["data_ready_for_modeling", "selected_features_per_target"],
                outputs=[
                    "roi_within_horizon_results",
                    "test_set_with_roi_within_horizon_preds",
                ],
            ),
        ]
    )
