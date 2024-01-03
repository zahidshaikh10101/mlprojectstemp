import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(param,models,X_train_array,y_train_array,X_test_array,y_test_array):
    try:
        score_dict={}
        for i in range(len(list(models))):
            model_name=list(models.keys())[i]
            model=list(models.values())[i]
            para=param[model_name]
            rf_model=GridSearchCV(model,para)
            rf_model.fit(X_train_array,y_train_array)
            model.set_params(**rf_model.best_params_)
            model.fit(X_train_array,y_train_array)
            y_pred=model.predict(X_test_array)
            model_score=r2_score(y_pred,y_test_array)
            score_dict[model_name]=model_score
        return score_dict
    except Exception as e:
        raise CustomException("Specific error message describing the issue", e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)