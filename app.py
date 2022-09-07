from arima.arima_model_train import Arima_Train
from arima.arima_model_predict import Arima_Predict
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from arima.arima_plots import Arima_Plots
from arima.constants import *
import os

app = Flask(__name__)


IMG_FOLDER = os.path.join(GRAPH_DIR)

app.config['UPLOAD_FOLDER'] = IMG_FOLDER
# plot = Arima_Plots()

# plot.plot_all()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train', methods=['GET'])
def train():
    train = Arima_Train()
    train.train()
    return render_template('train.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        fromDate = request.form.get("from")
        toDate = request.form.get("to")

        output = Arima_Predict.predict(datetime(int(toDate.split('-')[0]), int(toDate.split('-')[1]), int(toDate.split('-')[2])), datetime(int(fromDate.split('-')[0]),int(fromDate.split('-')[1]), int(fromDate.split('-')[2])))

        return render_template('result.html', outputs = output)

@app.route('/plots', methods=['POST'])
def plot():
    if request.method == 'POST':
        plot = Arima_Plots()
        plot.plot_all()

    data = os.path.join(app.config['UPLOAD_FOLDER'], 'data.png')
    rolling_values = os.path.join(app.config['UPLOAD_FOLDER'], 'rolling_values.png')
    acf = os.path.join(app.config['UPLOAD_FOLDER'], 'acf.png')
    decompose = os.path.join(app.config['UPLOAD_FOLDER'], 'decompose.png')
    return render_template('plots.html',DATA=data, ROLLING_VALUES=rolling_values, ACF=acf, DECOMPOSE=decompose)

if __name__=='__main__':
    app.run(debug=True)