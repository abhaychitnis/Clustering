from sklearn.cluster import KMeans

from sklearn import metrics
from sklearn.metrics import pairwise_distances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def best_model(X, plot_ind, eval_parm):
    parm_list = []
    if eval_parm == 'deep':
        parm_list.append({'n_clusters': 4, 'tol': 0.5e-4})
        parm_list.append({'n_clusters': 5, 'tol': 0.5e-4})
        parm_list.append({'n_clusters': 6, 'tol': 0.5e-4})
        parm_list.append({'n_clusters': 7, 'tol': 0.5e-4})
        parm_list.append({'n_clusters': 8, 'tol': 0.5e-4})
        parm_list.append({'n_clusters': 4, 'tol': 1e-4})
        parm_list.append({'n_clusters': 5, 'tol': 1e-4})
        parm_list.append({'n_clusters': 6, 'tol': 1e-4})
        parm_list.append({'n_clusters': 7, 'tol': 1e-4})
        parm_list.append({'n_clusters': 8, 'tol': 1e-4})
        parm_list.append({'n_clusters': 4, 'tol': 1.5e-4})
        parm_list.append({'n_clusters': 5, 'tol': 1.5e-4})
        parm_list.append({'n_clusters': 6, 'tol': 1.5e-4})
        parm_list.append({'n_clusters': 7, 'tol': 1.5e-4})
        parm_list.append({'n_clusters': 8, 'tol': 1.5e-4})
    elif eval_parm == 'test':
        parm_list.append({'n_clusters': 4})
        parm_list.append({'n_clusters': 5})
        parm_list.append({'n_clusters': 6})
        parm_list.append({'n_clusters': 7})
        parm_list.append({'n_clusters': 8})
    s_score_list = []
    ch_score_list = []
    kmeans = KMeans(init = 'k-means++', random_state = 42)
    for i in range(len(parm_list)):
        kmeans.set_params(**parm_list[i]).fit(X)
        labels = kmeans.labels_
        s_score_list.append \
                (metrics.silhouette_score(X, labels, metric='euclidean'))
        ch_score_list.append \
                (metrics.calinski_harabaz_score(X, labels))       
    s_score = s_score_list[np.argmax(s_score_list)]
    ch_score = ch_score_list[np.argmax(s_score_list)]

#    for i in range(len(s_score_list)):
#        print (parm_list[i], s_score_list[i], ch_score_list[i])

    kmeans.set_params(**parm_list[np.argmax(s_score_list)]).fit(X)

    print (parm_list[np.argmax(s_score_list)])                       

    return_parm= {'trained_model': kmeans, \
                  's_score': s_score, 'ch_score': ch_score}


    return (return_parm)
