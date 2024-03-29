import os
import sys
import dill
from src.exception import CustomException

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_object:
            dill.dump(obj, file_object)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            parameters = params[(list(models.keys()))[i]]

            grid = GridSearchCV(model, parameters, cv=3)
            grid.fit(X_train, y_train)

            model.set_params(**grid.best_params_)
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)

            # y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # training_score = r2_score(y_train, y_train_pred)
            testing_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = testing_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
