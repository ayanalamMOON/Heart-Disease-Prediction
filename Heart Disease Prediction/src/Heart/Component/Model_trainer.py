import os
import sys
import numpy as np
import pandas as pd 
from sklearn.svm import svc
from xgboost import XGBClassifier
from dataclasses import dataclass
from src.Hert.logger import logging
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from src.Heart.exception import customExeception
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.Heart.utils.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('Artifacts', 'Model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_traing(self, train_array, test_array)
        try:
            logging.info('Splitting Dependent and Independent variables from the tarin and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train-array[:, :-1],
                test_array[:, :-1],
                test_array[:, :-1]

            model = {
                'Logistic Regression': LogisticRegression(),
                'Naive Bayed': GaussianNB(),
                'Random Forest Classifier' : RandomForestClassifier(n_estimators=20, random_state=12, max_depth=5),
                'XG Boost': XGBClassifier(learning_rate=)

            }

            )


