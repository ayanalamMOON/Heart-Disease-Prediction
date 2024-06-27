import os
import sys
import mlflow
import pickle
import numpy as np
import pandas as pd 
import mlflow.sklearn
from urllib.parse import urlparse
from src.Heart.utils.utils import load_object
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluation: 
    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return accuracy, precision, recall , f1
    
    def initiate_model_evaluaton(self_train, train_array, test_array):
        try:
            X_test,y_test=(test_array[:,:-1], test_array[:.-1])
            model_path = os.path.join("Artifacts", "Model.pkl")
            model=load_object(model_path)

            mlflow.set_registary_uri("https://dagshub.com/HemaKalyan45/Heart-Disease-Prediction.mlflow")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            print(tracking_url_type_store)

            with mlflow.start_run():

                predicted_qualities = model.predict(X_test)

                (accuracy, precision, recall, f1) = self.eval_metrics(y_test, predicted_qualities)

                if tarcking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "Model", registered_model_name = "ml_model")
                else:
                    mlflow.sklearn.log_model(model, "Model")

            except Exception as e:
                raise customExeception(e, sys)

