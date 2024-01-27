import os
import sys
from dataclasses import dataclass


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model

@dataclass
class ModelTrainierConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainierConfig()

    def initiate_model_training(self, training_array, test_array):

        try:
            logging.info('Splitting training and test input data')
            X_train, y_train, X_test, y_test = (
                training_array[:, :-1],
                training_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
        
            models = {
                'LinearRegression': LinearRegression(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor()
            }

            model_report :dict = evaluate_model(X_train = X_train, y_train=y_train, X_test = X_test, 
                                                y_test = y_test, models = models)
            

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            
            logging.info('Best model found on both training and test dataset')

            save_obj(
                file_path= self.model_trainer_config.trained_model_path,
                obj= best_model
            )

            predicted = best_model.predict(X_test)
            r_square_score = r2_score(y_test, predicted)

            return r_square_score

        except Exception as e:
            raise CustomException(e, sys)