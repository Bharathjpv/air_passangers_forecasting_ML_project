# Air Passengers forecasting

This is a time series analysis problem when we predict the number of passengers for the upcoming months such that we can plan the number of flights required.

## 💽 Dataset
It is an open-source dataset on Kaggle which has the number of passengers month-wise for 10 years.

## 📚 Approach 
1. Do the indexing of the data such that it will be easy to visualize.
2. Use auto_arima from pmdarima library.
3. The same model is used for the forecasting

## 🧑‍💻 How to setup
create fresh conda environment 
```python
conda create -p venv python=3.7 -y
```
activate conda environment
```python
conda activate venv/
```
Install requirements
```python
pip install -r requirements.txt
```
Run the web app
```python
python app.py
```
To launch the ui
```python
http://localhost:5000/
```

## 🧑‍💻 Tech Used
1. Machine Learning
2. Time series analysis 
3. ARIMA 
4. Flask

## 🏭 Industrial Use-cases 
1. Forecasting the number of planes required for the coming month
2. It helps in the timely planning of any events in advance.

## 👋 Conclusion
Based on the forecast we can plan our upcoming month's planes and if done for individual routes we can even assign the available resources optimally.