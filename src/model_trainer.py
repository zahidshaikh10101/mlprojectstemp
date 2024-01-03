import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from exception import CustomException
from logger import logging

from utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        logging.info("Define the model saving path")


    def initiate_model_trainer(self,X_train_array,y_train_array,X_test_array,y_test_array):
        try:
            self.X_train_array = X_train_array
            self.y_train_array = y_train_array
            self.X_test_array = X_test_array
            self.y_test_array = y_test_array
            logging.info("Define the training and testing array for x and y")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "ExtraTree Regressor": ExtraTreesRegressor(),
            }
            logging.info("Models Are defined Successfully.")

            ##HypertuneParamter
            params={
                "Random Forest":{
                    'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    #'max_features':['sqrt','log2'],
                    'n_estimators': [8,16,32,64,128] 
                },
                "Decision Tree":{
                    'criterion':['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'splitter':['best','random'],
                    #'max_features':['sqrt','log2']
                },
                "Linear Regression":{},
                "ExtraTree Regressor":{
                    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    'n_estimators': [8,16,32,64,128],
                    #'max_depth': [1, 2, 5, 8, 13, 21, 34, 53, 54, 55, 89, None],
                    #'min_samples_split': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377],
                    #'min_samples_leaf': [1, 2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377],
                    'max_features': ['auto', 'sqrt', 'log2', 2, 5, 8, 13, 21, 34, None],
                    #'max_leaf_nodes': [2, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, None],
                }
            }
            logging.info("Hyperparameter completed")

            score_dict = evaluate_models(param=params,models=models,X_train_array=X_train_array,y_train_array=y_train_array,
                                       X_test_array=X_test_array,y_test_array=y_test_array)
            
            ## To get best model name from dict
            best_model_name = max(list(score_dict))

            ## To get best model score from dict
            best_model_score = score_dict[best_model_name]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model has been found and best model is {best_model_name} with test accuracy of {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test_array)

            r2_square = r2_score(y_test_array, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException("Specific error message", e, sys.exc_info())