import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    return start_date, end_date

def get_data_from_csv(start_date, end_date):
    # Read the CSV file
    df = pd.read_csv('sp500_stocks.csv')
    
    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter the dataframe based on the date range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    return df

def calculate_daily_returns(data):
    data['daily_return'] = data.groupby('Symbol')['Close'].pct_change()
    return data

def calculate_average_daily_return(data):
    average_daily_returns = data.groupby('Date')['daily_return'].mean().reset_index()
    return average_daily_returns

def show():
    # Get user input
    start_date, end_date = get_input()

    # Fetch data from the CSV file
    df = get_data_from_csv(start_date, end_date)

    # Calculate daily returns for each stock
    df = calculate_daily_returns(df)

    # Calculate the average daily return across all stocks
    average_daily_returns = calculate_average_daily_return(df)

    # Display the data and summary statistics side by side
    st.write(f"Showing average daily returns for US stocks from {start_date} to {end_date}")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(average_daily_returns)

    with col2:
        st.dataframe(average_daily_returns.describe())

    # Plotting average daily returns
    st.write(f"Average daily returns of stock universe chart from {start_date} to {end_date}")
    st.line_chart(average_daily_returns.set_index('Date')['daily_return'])

if __name__ == "__main__":
    show()
