import streamlit as st
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
    return start_date, end_date, stock_symbol

def get_data_from_db(stock_symbol, start_date, end_date):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    query = f"""
    SELECT date, open, close FROM reporting.mtd_daily_stock_data
    WHERE symbol = '{stock_symbol}'
    AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def add_normalized_column(data, column_name):
    scaler = MinMaxScaler()
    data[column_name + "_normalized"] = scaler.fit_transform(data[[column_name]])

def add_daily_return_column(data):
    data['daily_return'] = (data['open'] - data['close']) / data['close']


def show():
    # Get user input
    start_date, end_date, stock_symbol = get_input()

    # Fetch data from the database
    df = get_data_from_db(stock_symbol, start_date, end_date)

    # Display the data and summary statistics side by side
    st.write(f"Showing data and summary statistics for {stock_symbol} from {start_date} to {end_date}")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df)

    with col2:
        st.dataframe(df.describe())

    # Plotting closed
    st.write(f"Chart for {stock_symbol} from {start_date} to {end_date}")
    st.line_chart(df.set_index('date')['close'])

    # Adding normalized close column
    add_normalized_column(df, 'close')

    # Plotting normalied closed
    st.write(f"Chart for {stock_symbol} from {start_date} to {end_date} normalized")
    st.line_chart(df.set_index('date')['close_normalized'])

    # Adding daily return from close
    add_daily_return_column(df)

    # Plotting daily returns
    st.write(f"Daily returns chart for {stock_symbol} from {start_date} to {end_date} normalized")
    st.line_chart(df.set_index('date')['daily_return'])


