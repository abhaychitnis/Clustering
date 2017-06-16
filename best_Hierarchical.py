from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics
from sklearn.metrics import pairwise_distances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def best_model(X, plot_ind, eval_parm):
    parm_list = []
    if eval_parm == 'deep':
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'euclidian', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'euclidian', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'euclidian', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'l1', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'l1', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'l1', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'l2', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'l2', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'l2', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'cosine', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'cosine', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'manhattan', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'manhattan', \
                          'linkage': 'average'})

        parm_list.append({'n_clusters': 5, \
                        'affinity': 'euclidian', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'euclidian', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'euclidian', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'l1', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'l1', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'l1', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'l2', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'l2', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'l2', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'cosine', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'cosine', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'manhattan', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'manhattan', \
                          'linkage': 'average'})


        parm_list.append({'n_clusters': 6, \
                        'affinity': 'euclidian', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'euclidian', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'euclidian', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'l1', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'l1', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'l1', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'l2', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'l2', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'l2', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'cosine', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'cosine', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'manhattan', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'manhattan', \
                          'linkage': 'average'})

        parm_list.append({'n_clusters': 7, \
                        'affinity': 'euclidian', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'euclidian', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'euclidian', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'l1', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'l1', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'l1', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'l2', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'l2', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'l2', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'cosine', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'cosine', \
                          'linkage': 'average'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'manhattan', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'manhattan', \
                          'linkage': 'average'})



    elif eval_parm == 'test':
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'euclidean', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 4, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})

        parm_list.append({'n_clusters': 5, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'euclidean', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 5, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})

        parm_list.append({'n_clusters': 6, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'euclidean', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 6, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})

        parm_list.append({'n_clusters': 7, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'euclidean', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 7, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})

        parm_list.append({'n_clusters': 8, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 8, \
                        'affinity': 'euclidean', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 8, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 8, \
                        'affinity': 'cosine', \
                          'linkage': 'complete'})
        parm_list.append({'n_clusters': 8, \
                        'affinity': 'euclidean', \
                          'linkage': 'ward'})
        parm_list.append({'n_clusters': 8, \
                        'affinity': 'manhattan', \
                          'linkage': 'complete'})


    s_score_list = []
    ch_score_list = []
    hc = AgglomerativeClustering()
    for i in range(len(parm_list)):
        #print (parm_list[i])
        hc.set_params(**parm_list[i]).fit(X)
        labels = hc.labels_
        s_score_list.append \
                (metrics.silhouette_score(X, labels, metric='euclidean'))
        ch_score_list.append \
                (metrics.calinski_harabaz_score(X, labels))       
    s_score = s_score_list[np.argmax(s_score_list)]
    ch_score = ch_score_list[np.argmax(s_score_list)]

#    for i in range(len(s_score_list)):
#        print (parm_list[i], s_score_list[i], ch_score_list[i])

    print (parm_list[np.argmax(s_score_list)])
    hc.set_params(**parm_list[np.argmax(s_score_list)]).fit(X)
                        
    #s_score = metrics.silhouette_score(X, labels, metric='euclidean')
    #ch_score = metrics.calinski_harabaz_score(X, labels)

    return_parm= {'trained_model': hc, \
                  's_score': s_score, 'ch_score': ch_score}


    return (return_parm)
