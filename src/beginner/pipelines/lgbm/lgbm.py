import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import optuna
from optuna.integration.lightgbm import LightGBMTunerCV as tuner

def cross_validation_model(model_params, train_x, train_y, k, seed):
    logger = logging.getLogger(__name__)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    model_params["seed"] = seed

    accuracys = []

    for train_idx, val_idx in kf.split(train_x):
        train_data = lgb.Dataset(train_x.iloc[train_idx], label=train_y[train_idx])
        valid_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])

        lgbm = lgb.train(model_params, train_data, valid_sets=valid_data,)

        preds = lgbm.predict(train_x.iloc[val_idx])
        accuracy = accuracy_score(preds.argmax(axis=1), train_y[val_idx])
        accuracys.append(accuracy)

    accuracys = np.array(accuracys)
    accuracy = accuracys.mean()
    logger.info("accuracy:%f", accuracy)
    return accuracy

def hyper_parameter_tuning(model_params, train_x, train_y, k, seed, tuning_params):
    train_data = lgb.Dataset(train_x, label=train_y)
    booster=tuner(model_params,train_data,nfold=k,seed=seed)
    booster.run()
    print(booster.best_score)
    print(booster.best_params)
    return booster

def train(model_params, train_x, train_y, seed, train_rate):
    logger = logging.getLogger(__name__)

    model_params["seed"] = seed

    data_len = len(train_x)
    train_idx = np.ones(data_len, dtype=bool)
    train_idx[int(data_len * train_rate) :] = False
    np.random.seed(seed)
    np.random.shuffle(train_idx)
    val_idx = train_idx == False

    train_data = lgb.Dataset(train_x[train_idx], label=train_y[train_idx])
    valid_data = lgb.Dataset(train_x[val_idx], label=train_y[val_idx])

    lgbm = lgb.train(model_params, train_data, valid_sets=valid_data)

    preds = lgbm.predict(train_x.iloc[val_idx])
    accuracy = accuracy_score(preds.argmax(axis=1), train_y[val_idx])
    logger.info("accuracy:%f", accuracy)

    return lgbm


def predict(test_data, lgbm, test_data_index):
    preds = lgbm.predict(test_data)
    return pd.DataFrame([test_data_index, preds.argmax(axis=1)]).T

def pseudo_label(test_data ,lgbm, th, test_data_index):
    preds = lgbm.predict(test_data)
    df=pd.DataFrame([test_data_index, preds.argmax(axis=1),preds.max(axis=1)]).T
    df.columns=["index","predict","score"]
    print(df.sort_values("score"))
    df=df.sort_values("index")
    df=df.loc[df["score"]>th]
    print(df)
    pseudo_data=test_data.loc[df["index"]]
    pseudo_data["price_range"]=df["predict"].to_numpy()
    return pseudo_data

def select_feature(lgbm, params, train_x, train_y):
    selector = SelectFromModel(lgb.LGBMClassifier(**params), threshold="median")
    selector.fit(train_x, train_y)
    print(train_x.columns)
    print(selector.get_support())
    print(train_x.columns[selector.get_support()])
    print(train_x.columns[selector.get_support()==False])
    print(selector.estimator_.feature_importances_)
    return {
            "columns":train_x.columns,
            "importances":selector.estimator_.feature_importances_,
            "feature_selected":train_x.columns[selector.get_support()],
            "feature_not_selected":train_x.columns[selector.get_support()==False],
        }