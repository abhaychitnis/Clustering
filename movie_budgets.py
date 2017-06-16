
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


preprocessing_override = pd.read_csv('movie_budgets_override.csv')

# Importing the dataset
dataset_X = pd.read_csv('movie_budgets.csv')


dataset_X = dataset_X.ix[:,4:7]
print (dataset_X.head(5))
preprocessing_override = preprocessing_override.ix[:, 4:7]


import preprocess_data as prd

preprocessed_data = prd.preprocess_cluster_data(dataset_X, \
                        preprocessing_override)

X = preprocessed_data["X"]
#X = dataset_X.iloc[:, :].values

#print (dataset_X)
#print (X)

# import get_best_model as bfm
import get_best_clustering_model as bfm

best_fit_model = bfm.get_best_model(X)

#import plot_best_model as pbm

#pbm.plot_best_model(X,best_fit_model)

#print (best_fit_model)
