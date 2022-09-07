import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import os

from arima.constants import *

class Arima_Plots:
    def __init__(self):

        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        self.plot_path = os.path.join(ARTIFACT_DIR, GRAPH_DIR)
        os.makedirs(self.plot_path, exist_ok=True)
        self.data_path = DATA_PATH
        self.df = pd.read_csv(self.data_path)
        self.df['Month'] = pd.to_datetime(self.df['Month'], format='%Y-%m')
        self.df.index = self.df['Month']
        del self.df['Month']        
    
    def plot_data(self):
        plt.plot(self.df)
        plt.ylabel('Number of Passengers')
        plt.savefig(os.path.join(self.plot_path, 'data.png'))
    
    def plot_rolling_values(self):
        rolling_mean = self.df.rolling(7).mean()
        rolling_std = self.df.rolling(7).std()
        plt.plot(self.df, color='blue',label='Original Passenger Data')
        plt.plot(rolling_mean, color='red', label='Rolling Mean Passenger Number')
        plt.plot(rolling_std, color='black', label = 'Rolling Standard Deviation in Passenger Number')
        plt.legend(loc='best');
        plt.savefig(os.path.join(self.plot_path, 'rolling_values.png'))

    def plot_autocorrelation(self):
        plot_acf(self.df)
        plt.savefig(os.path.join(self.plot_path, 'acf.png'))
    
    def plot_decomposition(self):
        decompose = seasonal_decompose(self.df['#Passengers'],model='additive', period=7)
        decompose.plot()
        plt.savefig(os.path.join(self.plot_path, 'decompose.png'))

    def plot_all(self):
        self.plot_data()
        self.plot_rolling_values()
        self.plot_autocorrelation()
        self.plot_decomposition()
