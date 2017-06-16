from sklearn.cluster import Birch

from sklearn import metrics
from sklearn.metrics import pairwise_distances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def best_model(X, plot_ind, eval_parm):
    parm_list = []
    if eval_parm == 'deep':
        parm_list.append({'threshold': 0.3, 'branching_factor': 30})
        parm_list.append({'threshold': 0.3, 'branching_factor': 40})
        parm_list.append({'threshold': 0.3, 'branching_factor': 50})
        parm_list.append({'threshold': 0.3, 'branching_factor': 60})
        parm_list.append({'threshold': 0.3, 'branching_factor': 80})
        parm_list.append({'threshold': 0.4, 'branching_factor': 30})
        parm_list.append({'threshold': 0.4, 'branching_factor': 40})
        parm_list.append({'threshold': 0.4, 'branching_factor': 50})
        parm_list.append({'threshold': 0.4, 'branching_factor': 60})
        parm_list.append({'threshold': 0.4, 'branching_factor': 80})
        parm_list.append({'threshold': 0.5, 'branching_factor': 30})
        parm_list.append({'threshold': 0.5, 'branching_factor': 40})
        parm_list.append({'threshold': 0.5, 'branching_factor': 50})
        parm_list.append({'threshold': 0.5, 'branching_factor': 60})
        parm_list.append({'threshold': 0.5, 'branching_factor': 80})
        parm_list.append({'threshold': 0.6, 'branching_factor': 30})
        parm_list.append({'threshold': 0.6, 'branching_factor': 40})
        parm_list.append({'threshold': 0.6, 'branching_factor': 50})
        parm_list.append({'threshold': 0.6, 'branching_factor': 60})
        parm_list.append({'threshold': 0.6, 'branching_factor': 80})
        parm_list.append({'threshold': 0.7, 'branching_factor': 30})
        parm_list.append({'threshold': 0.7, 'branching_factor': 40})
        parm_list.append({'threshold': 0.7, 'branching_factor': 50})
        parm_list.append({'threshold': 0.7, 'branching_factor': 60})
        parm_list.append({'threshold': 0.7, 'branching_factor': 80})
    elif eval_parm == 'test':
        parm_list.append({'threshold': 0.4, 'branching_factor': 40})
        parm_list.append({'threshold': 0.4, 'branching_factor': 50})
        parm_list.append({'threshold': 0.4, 'branching_factor': 60})
        parm_list.append({'threshold': 0.5, 'branching_factor': 40})
        parm_list.append({'threshold': 0.5, 'branching_factor': 50})
        parm_list.append({'threshold': 0.5, 'branching_factor': 60})
        parm_list.append({'threshold': 0.6, 'branching_factor': 40})
        parm_list.append({'threshold': 0.6, 'branching_factor': 50})
        parm_list.append({'threshold': 0.6, 'branching_factor': 60})
    s_score_list = []
    ch_score_list = []
    br = Birch()
    for i in range(len(parm_list)):
        br.set_params(**parm_list[i]).fit(X)
        labels = br.labels_
        s_score_list.append \
                (metrics.silhouette_score(X, labels, metric='euclidean'))
        ch_score_list.append \
                (metrics.calinski_harabaz_score(X, labels))       

#    for i in range(len(parm_list)):
#        print (parm_list[i], s_score_list[i], ch_score_list[i])

    s_score = s_score_list[np.argmax(s_score_list)]
    ch_score = ch_score_list[np.argmax(s_score_list)]


    br.set_params(**parm_list[np.argmax(s_score_list)]).fit(X)

    print (parm_list[np.argmax(s_score_list)])
    print (len(br.subcluster_centers_ ))

    return_parm= {'trained_model': br, \
                  's_score': s_score, 'ch_score': ch_score}


    return (return_parm)
