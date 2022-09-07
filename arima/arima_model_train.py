import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from arima import logger
from arima.utils import save_model
from arima.constants import *





class Arima_Train:

    def __init__(self):
        self.data_path = DATA_PATH

    def get_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df['Month'] = pd.to_datetime(self.df['Month'], format='%Y-%m')
        logger.info('Data is ready for training')
    

    def train_test_split(self):
        self.train = self.df[self.df['Month'] < '1960-08-01']
        self.train['train'] = self.train['#Passengers']
        del self.train['Month']
        del self.train['#Passengers']
        self.test = self.df[self.df['Month'] >= '1960-08-01']
        del self.test['Month']
        self.test['test'] = self.test['#Passengers']
        del self.test['#Passengers']
        logger.info('Train test split done')
    
    def fit_arima(self):
        self.model = auto_arima(self.train, trace=True, error_action='ignore', suppress_warnings=True)
        self.model.fit(self.train)
        logger.info(f'model is fit and the best model is {self.model.order} and seasonal order is {self.model.seasonal_order}')

    def test_arima(self):
        predict = self.model.predict(n_periods=len(self.test))
        self.predict = pd.DataFrame(predict,index = self.test.index,columns=['Prediction'])

        train_rmse = ((self.model.resid()**2).mean())**(1/2)
        test_rmse = (mean_squared_error(self.test, self.predict))

        print(train_rmse, test_rmse)
        logger.info(f'testing of the model is done with Train RMSE is {train_rmse} and Test RMSE is {test_rmse}')

        return train_rmse, test_rmse
    
    def train(self):
        self.get_data()
        self.train_test_split()
        self.fit_arima()
        tr_rmse, tst_rmse = self.test_arima()

        os.makedirs(os.path.join(ROOT, ARTIFACT_DIR), exist_ok=True)

        save_model(self.model, os.path.join(ROOT, ARTIFACT_DIR, MODEL_PATH))
        logger.info('model saved')



