from sklearn.cluster import AffinityPropagation

from sklearn import metrics
from sklearn.metrics import pairwise_distances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def best_model(X, plot_ind, eval_parm):
    parm_list = []
    if eval_parm == 'deep':
        parm_list.append({'damping': 0.5, 'convergence_iter': 14})
        parm_list.append({'damping': 0.5, 'convergence_iter': 15})
        parm_list.append({'damping': 0.5, 'convergence_iter': 16})
        parm_list.append({'damping': 0.5, 'convergence_iter': 20})
        parm_list.append({'damping': 0.5, 'convergence_iter': 25})
        parm_list.append({'damping': 0.75, 'convergence_iter': 14})
        parm_list.append({'damping': 0.75, 'convergence_iter': 15})
        parm_list.append({'damping': 0.75, 'convergence_iter': 16})
        parm_list.append({'damping': 0.75, 'convergence_iter': 20})
        parm_list.append({'damping': 0.75, 'convergence_iter': 25})
        parm_list.append({'damping': 0.99, 'convergence_iter': 14})
        parm_list.append({'damping': 0.99, 'convergence_iter': 15})
        parm_list.append({'damping': 0.99, 'convergence_iter': 16})
        parm_list.append({'damping': 0.99, 'convergence_iter': 20})
        parm_list.append({'damping': 0.99, 'convergence_iter': 25})
    elif eval_parm == 'test':
        parm_list.append({'damping': 0.5, 'convergence_iter': 15})
        parm_list.append({'damping': 0.5, 'convergence_iter': 20})
        parm_list.append({'damping': 0.5, 'convergence_iter': 25})
        parm_list.append({'damping': 0.75, 'convergence_iter': 15})
        parm_list.append({'damping': 0.75, 'convergence_iter': 20})
        parm_list.append({'damping': 0.75, 'convergence_iter': 25})
        parm_list.append({'damping': 0.90, 'convergence_iter': 15})
        parm_list.append({'damping': 0.90, 'convergence_iter': 20})
        parm_list.append({'damping': 0.90, 'convergence_iter': 25})
    s_score_list = []
    ch_score_list = []
    ap = AffinityPropagation(preference=-50)
    for i in range(len(parm_list)):
        ap.set_params(**parm_list[i]).fit(X)
        labels = ap.labels_
        s_score_list.append \
                (metrics.silhouette_score(X, labels, metric='euclidean'))
        ch_score_list.append \
                (metrics.calinski_harabaz_score(X, labels))       

#    for i in range(len(parm_list)):
#        print (parm_list[i], s_score_list[i], ch_score_list[i])

    s_score = s_score_list[np.argmax(s_score_list)]
    ch_score = ch_score_list[np.argmax(s_score_list)]


    ap.set_params(**parm_list[np.argmax(s_score_list)]).fit(X)

    print (parm_list[np.argmax(s_score_list)])
    print (len(ap.cluster_centers_indices_))

    return_parm= {'trained_model': ap, \
                  's_score': s_score, 'ch_score': ch_score}


    return (return_parm)
