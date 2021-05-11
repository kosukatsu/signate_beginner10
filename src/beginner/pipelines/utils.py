import pickle

import pandas as pd


def split_data(data):
    y = data["price_range"].to_numpy()
    x = data.drop("price_range", axis=1).copy()
    return x, y

def drop_feature(data,feature_dropping):
    return data.drop(feature_dropping,axis=1).copy()

def merge_dictionary(dict1,dict2):
    dict1.update(dict2)
    return dict1

def merge_pseudo_data(data1,data2,use_pseudo):
    if use_pseudo:
        df=pd.concat([data1,data2])
        df["price_range"]=df["genreprice_range"].astype(int)
        return df
    else:
        return data1

def get_test_index(data):
    return data[0].to_numpy()
