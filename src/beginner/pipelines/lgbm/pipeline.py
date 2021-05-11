import os, sys

sys.path.append("./src/beginner/pipelines")

from kedro.pipeline import Pipeline, node

from .lgbm import (
    cross_validation_model,
    hyper_parameter_tuning,
    train,
    predict,
    select_feature,
    pseudo_label,
)
from utils import split_data,drop_feature,merge_dictionary, merge_pseudo_data


def create_cross_validation_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                drop_feature,
                ["train_data_set","params:drop_feature"],
                "train_data_dropped_feature"
            ),
            node(
                merge_pseudo_data,
                ["train_data_dropped_feature","pseudo_data","params:use_pseudo_label"],
                "train_data_concated_pseudo"
            ),
            node(split_data, "train_data_concated_pseudo", ["train_x", "train_y"],),
            node(
                merge_dictionary,
                ["params:default_lgbm_params","params:lgbm_params"],
                "lgbm_params"
            ),
            node(
                cross_validation_model,
                {
                    "model_params": "lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "k": "params:cross_validation_k",
                    "seed": "params:seed",
                },
                "accuracy",
            ),
        ]
    )


def create_hy_para_tuning_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                drop_feature,
                ["train_data_set","params:drop_feature"],
                "train_data_dropped_feature"
            ),
            node(
                merge_pseudo_data,
                ["train_data_dropped_feature","pseudo_data","params:use_pseudo_label"],
                "train_data_concated_pseudo"
            ),
            node(split_data, "train_data_concated_pseudo", ["train_x", "train_y"],),
            node(
                hyper_parameter_tuning,
                {
                    "model_params": "params:default_lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "k": "params:cross_validation_k",
                    "seed": "params:seed",
                    "tuning_params": "params:lgbm_hyper_parameter_tuning",
                },
                "lgbm_model_hypara_tuning",
            ),
        ]
    )


def create_real_train_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                merge_dictionary,
                ["params:default_lgbm_params","params:lgbm_params"],
                "lgbm_params"
            ),
            node(
                drop_feature,
                ["train_data_set","params:drop_feature"],
                "train_data_dropped_feature"
            ),
            node(
                merge_pseudo_data,
                ["train_data_dropped_feature","pseudo_data","params:use_pseudo_label"],
                "train_data_concated_pseudo"
            ),
            node(split_data, "train_data_concated_pseudo", ["train_x", "train_y"],),
            node(
                train,
                {
                    "model_params": "lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "seed": "params:seed",
                    "train_rate": "params:train_rate",
                },
                "lgbm_model",
            ),
        ],
    )


def create_eval_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                drop_feature,
                ["test_data_set","params:drop_feature"],
                "test_data_dropped_feature"
                ),
            node(
                predict, 
                ["test_data_dropped_feature", "lgbm_model"], 
                "lgbm_output"),
        ],
    )

def create_pseudo_label_pipeline(**kwargs):
    return Pipeline([
        node(
            drop_feature,
            ["test_data_set","params:drop_feature"],
            "test_data_dropped_feature"
            ),
        node(
            pseudo_label, 
            ["test_data_dropped_feature", "lgbm_model","params:importance_threthold"], 
            "pseudo_data"
            ),
        ]
        )

def create_select_feature_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                merge_dictionary,
                ["params:default_lgbm_params","params:lgbm_params"],
                "lgbm_params"
                ),
            node(
                drop_feature,
                ["train_data_set","params:drop_feature"],
                "train_data_dropped_feature"
                ),
            node(split_data, "train_data_dropped_feature", ["train_x", "train_y"]),
            # node(
            #     merge_pseudo_data,
            #     ["train_data_dropped_feature","pseudo_data","params:use_pseudo_label"],
            #     "train_data_concated_pseudo"
            #     ),
            node(
                select_feature,
                ["lgbm_model", "lgbm_params", "train_x", "train_y"],
                "feature_importance",
            ),
        ]
    )