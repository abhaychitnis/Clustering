import pandas as pd
import pickle


def get_best_model(X):


    import importlib
    model_to_evaluate = pd.read_csv('clustering_models_list.csv')

    return_list = []
    best_score = 0
    best_model = None
    for i in range(len(model_to_evaluate.index)):

        eval_parm = model_to_evaluate['eval_parm'][i]
    
        brf = importlib.import_module(model_to_evaluate['file_name'][i])
        return_parm = brf.best_model(X, False, eval_parm)
        
        return_parm["model_name"] = model_to_evaluate['model_name'][i]

        return_list.append(return_parm)

    best_model_found = evalModelList(return_list)
 
    return(pickle.dumps(best_model_found))

def evalModelList (return_list):

    modelStore = pd.DataFrame(return_list)

    print (modelStore)

    bestModelRow = modelStore.ix[modelStore['s_score'].idxmax()]

    print (bestModelRow)

    best_model_found = bestModelRow['trained_model']

    return(best_model_found)
