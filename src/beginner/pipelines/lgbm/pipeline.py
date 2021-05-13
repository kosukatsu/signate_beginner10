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
    dump_params,
)
from utils import split_data,drop_feature,merge_dictionary, merge_pseudo_data,get_test_index


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
                ["params:default_lgbm_params","tuning_params"],
                "lgbm_params",
            ),
            node(
                cross_validation_model,
                {
                    "model_params": "lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "k": "params:cross_validation_k",
                    "seed": "params:seed",
                    "class_label":"params:class_labels",
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
                "train_data_dropped_feature",
            ),
            node(
                merge_pseudo_data,
                ["train_data_dropped_feature","pseudo_data","params:use_pseudo_label"],
                "train_data_concated_pseudo",
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
                },
                "lgbm_model_hypara_tuning",
            ),
            node(
                dump_params,
                "lgbm_model_hypara_tuning",
                "tuning_params",
            ),
        ]
    )


def create_real_train_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                merge_dictionary,
                ["params:default_lgbm_params","tuning_params"],
                "lgbm_params",
            ),
            node(
                drop_feature,
                ["train_data_set","params:drop_feature"],
                "train_data_dropped_feature",
            ),
            node(
                merge_pseudo_data,
                ["train_data_dropped_feature","pseudo_data","params:use_pseudo_label"],
                "train_data_concated_pseudo",
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
                    "class_label":"params:class_labels",
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
                "test_data_dropped_feature",
                ),
            node(
                get_test_index,
                ["sample_submission"],
                "test_data_index",
            ),
            node(
                predict, 
                ["test_data_dropped_feature", "lgbm_model","test_data_index"], 
                "lgbm_output",
            ),
        ],
    )

def create_pseudo_label_pipeline(**kwargs):
    return Pipeline([
            node(
                drop_feature,
                ["test_data_set","params:drop_feature"],
                "test_data_dropped_feature",
                ),
            node(
                get_test_index,
                ["sample_submission"],
                "test_data_index",
            ),
            node(
                pseudo_label, 
                ["test_data_dropped_feature", "lgbm_model","params:importance_threthold","test_data_index"], 
                "pseudo_data",
            ),
        ]
        )

def create_select_feature_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                merge_dictionary,
                ["params:default_lgbm_params","tuning_params"],
                "lgbm_params",
                ),
            node(
                drop_feature,
                ["train_data_set","params:drop_feature"],
                "train_data_dropped_feature",
                ),
            node(split_data, "train_data_dropped_feature", ["train_x", "train_y"]),
            # node(
            #     merge_pseudo_data,
            #     ["train_data_dropped_feature","pseudo_data","params:use_pseudo_label"],
            #     "train_data_concated_pseudo"
            #     ),
            node(
                select_feature,
                ["lgbm_params", "train_x", "train_y"],
                "feature_importance",
            ),
        ]
    )