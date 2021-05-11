import pickle

import pandas as pd
import numpy as np

def typing(data):
    float_value_columns=["clock_speed","m_dep"]
    int_value_columns=["battery_power","fc","int_memory","mobile_wt","n_cores","pc","px_height","px_width","ram","sc_h","sc_w","talk_time"]
    bool_columns=["blue","dual_sim","four_g","three_g","touch_screen","wifi"]
    cate_columns=["price_range"]

    df[float_value_columns]=df[float_value_columns].astype(float)
    df[int_value_columns]=df[int_value_columns].astype(int)
    df[bool_columns]=df[bool_columns].astype(bool)
    df[cate_columns]=df[cate_columns].astype("category")
    
    return data
