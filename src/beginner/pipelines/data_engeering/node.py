import pickle

import pandas as pd
import numpy as np

def typing(data):
    float_value_columns=["clock_speed","m_dep"]
    int_value_columns=["battery_power","fc","int_memory","mobile_wt","n_cores","pc","px_height","px_width","ram","sc_h","sc_w","talk_time"]
    bool_columns=["blue","dual_sim","four_g","three_g","touch_screen","wifi"]

    data[float_value_columns]=data[float_value_columns].astype(float)
    data[int_value_columns]=data[int_value_columns].astype(int)
    data[bool_columns]=data[bool_columns].astype(bool)
    
    return data

def typing_target(data):
    cate_columns=["price_range"]
    data[cate_columns]=data[cate_columns].astype("category")
    return data