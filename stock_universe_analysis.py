import streamlit as st
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    return start_date, end_date

def get_data_from_db(start_date, end_date):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    query = f"""
    SELECT date, symbol, close FROM reporting.mtd_daily_stock_data
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date, symbol
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def calculate_daily_returns(data):
    data['daily_return'] = data.groupby('symbol')['close'].pct_change()
    return data

def calculate_average_daily_return(data):
    average_daily_returns = data.groupby('date')['daily_return'].mean().reset_index()
    return average_daily_returns

def show():
    # Get user input
    start_date, end_date = get_input()

    # Fetch data from the database
    df = get_data_from_db(start_date, end_date)

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
    st.line_chart(average_daily_returns.set_index('date')['daily_return'])

