import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import os

from arima.constants import *
from arima import logger

class Arima_Plots:
    def __init__(self):
        try:
            os.makedirs(GRAPH_DIR, exist_ok=True)
            self.data_path = DATA_PATH
            self.df = pd.read_csv(self.data_path)
            self.df['Month'] = pd.to_datetime(self.df['Month'], format='%Y-%m')
            self.df.index = self.df['Month']
            del self.df['Month']
            logger.info('Data preparation for plotting done')
        except Exception as e:
            logger.error('Plotting instance failed')
            raise e
    
    def plot_data(self):
        try:
            plt.plot(self.df)
            plt.ylabel('Number of Passengers')
            plt.savefig(os.path.join(GRAPH_DIR, 'data.png'))
            logger.info('data plot done')
        except Exception as e:
            logger.error('plotting the data failed')
            raise e
    
    def plot_rolling_values(self):
        try:
            rolling_mean = self.df.rolling(7).mean()
            rolling_std = self.df.rolling(7).std()
            plt.plot(self.df, color='blue',label='Original Passenger Data')
            plt.plot(rolling_mean, color='red', label='Rolling Mean Passenger Number')
            plt.plot(rolling_std, color= 'black', label = 'Rolling Standard Deviation in Passenger Number')
            plt.legend(loc='best');
            plt.savefig(os.path.join(GRAPH_DIR, 'rolling_values.png'))
            logger.info('rolling values plot done')
        except Exception as e:
            logger.error('rolling valaue plots failed')
            raise e

    def plot_autocorrelation(self):
        try:
            plot_acf(self.df)
            plt.savefig(os.path.join(GRAPH_DIR, 'acf.png'))
            logger.info('autocorrelation plot done')
        except Exception as e:
            logger.error('ACF plot failed')
            raise e
    
    def plot_decomposition(self):
        try:
            decompose = seasonal_decompose(self.df['#Passengers'],model='additive', period=7)
            decompose.plot()
            plt.savefig(os.path.join(GRAPH_DIR, 'decompose.png'))
            logger.info('decomposition plot done')
        except Exception as e:
            logger.error('decomposition plot failed')
            raise e

    def plot_all(self):
        try:
            self.plot_data()
            self.plot_rolling_values()
            self.plot_autocorrelation()
            self.plot_decomposition()
            logger.info('Plotting done')
        except Exception as e:
            logger.error('plotting failed')
            raise e
