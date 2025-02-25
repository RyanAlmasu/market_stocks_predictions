import streamlit as st
from neuralprophet import NeuralProphet
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import numpy as np

def get_data(ticker, interval='1d'):
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=50)).strftime('%Y-%m-%d')
    stock = yf.Ticker(ticker)
    company_name = stock.info.get("longName",ticker)
    data = stock.history(start=start, end=end, interval=interval)
    
    data = data[['Close']].reset_index()
    data.columns = ['ds','y'] 
    
    return data, company_name
    
def train_model(data,periods=30, test_size=0.2):
    
    #we gonna make a split 
    split_idx = int(len(data) * ( 1 - test_size))
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    model = NeuralProphet()
    model.fit(train_data, freq='D')
    
    future = model.make_future_dataframe(data, periods=periods)
    forecast = model.predict(future)
    
    test_pred = model.predict(test_data)['yhat1']
    rmse = np.sqrt(mean_squared_error(test_data['y'],test_pred))
    
    return forecast,rmse, train_data, test_data
    
def plot_forecast(train_data, test_data, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_data['ds'], y=train_data['y'],mode ='lines',name='Train Data'))
    fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['y'],mode ='lines',name='Test Data'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'],mode ='lines',name='Forecast'))
    return fig
    
def plot_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'],y=data['y'], mode='lines', name='Historical Data'))
    return fig

def main():
    st.title("Stock Price Prediction with NeuralProphet")
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="BTC-USD")
    period_slider = st.sidebar.slider("Prediction Days", min_value=7, max_value=14, value=7)
    period_input = st.sidebar.number_input("Or enter specific days", min_value=7, max_value=14) 
    period = period_input if period_input else period_slider
    
    
    st.write("Fetching stock data...")
    data, company_name = get_data(ticker)
    st.write(f'Latest Data for {company_name}:', data.tail())
    
    st.write("### Historical Prices")
    st.plotly_chart(plot_data(data))
    st.write(f"Selected Prediction Days: {period}")
    if st.button("Train Model"):
        st.write("Training NeuralProphet Model....")
        forecast, rmse , train_data, test_data = train_model(data, periods=period)
        st.write(f"### RMSE: {rmse:.2f}")
        
        st.write("### Forecasted Prices")
        st.plotly_chart(plot_forecast(train_data,test_data,forecast))

    
if __name__ == "__main__":
    main()
