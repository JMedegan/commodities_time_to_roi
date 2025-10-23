"""Project pipelines."""

from kedro.pipeline import Pipeline
from pipelines.time_to_roi.pipeline import create_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    time_to_roi = create_pipeline()

    return {
        "time_to_roi": time_to_roi,
    }
