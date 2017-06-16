import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def preprocess_cluster_data(dataset_X, override):

# common parameters

    missing_values_not_applicable = 0
    missing_values_drop_rows = 1
    missing_values_fill_mean = 2
    missing_values_drop_column = 3
    missing_values_not_decided = 4

    import common_functions as cm

    preprocess_list = cm.preprocess_ind(dataset_X, override)

    category_encoding_columns = preprocess_list[0]
    missing_values_strategy = preprocess_list[1]
    drop_strategy_columns = preprocess_list[2]
    normalize_strategy_columns = preprocess_list[3]


    X = dataset_X.iloc[:, :].values

    X_y_missing_values_managed = cm.manage_missing_values(X, None, missing_values_strategy)

    X = X_y_missing_values_managed['X']

    X = cm.manage_normalize_values(X, normalize_strategy_columns)
    
    X = cm.manage_category_encoding(X, category_encoding_columns, drop_strategy_columns)

    return ({"X":X})
