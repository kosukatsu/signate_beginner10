import os, sys

sys.path.append("./src/beginner/pipelines")

from .node import (
    typing,
    typing_target,
)

from kedro.pipeline import Pipeline, node

def create_test_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                typing,
                "test_raw_data_set",
                "test_data_set",
            )
        ]
    )
def create_train_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                typing,
                "train_raw_data_set",
                "train_typed",
            ),
            node(
                typing_target,
                "train_typed",
                "train_data_set",
            )
        ]
    )