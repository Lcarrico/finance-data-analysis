import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
    return start_date, end_date, stock_symbol

def get_data_from_csv(stock_symbol, start_date, end_date):
    # Read the CSV file
    df = pd.read_csv('sp500_stocks.csv')
    
    # Filter the dataframe based on user input
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    df = df[(df['Symbol'] == stock_symbol) & (df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    return df

def add_normalized_column(data, column_name):
    scaler = MinMaxScaler()
    data[column_name + "_normalized"] = scaler.fit_transform(data[[column_name]])

def add_daily_return_column(data):
    data['daily_return'] = (data['Open'] - data['Close']) / data['Close']

def show():
    # Get user input
    start_date, end_date, stock_symbol = get_input()

    # Fetch data from the CSV file
    df = get_data_from_csv(stock_symbol, start_date, end_date)

    # Display the data and summary statistics side by side
    st.write(f"Showing data and summary statistics for {stock_symbol} from {start_date} to {end_date}")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df)

    with col2:
        st.dataframe(df.describe())

    # Plotting close prices
    st.write(f"Chart for {stock_symbol} from {start_date} to {end_date}")
    st.line_chart(df.set_index('Date')['Close'])

    # Adding normalized close column
    add_normalized_column(df, 'Close')

    # Plotting normalized close prices
    st.write(f"Chart for {stock_symbol} from {start_date} to {end_date} normalized")
    st.line_chart(df.set_index('Date')['Close_normalized'])

    # Adding daily return from close
    add_daily_return_column(df)

    # Plotting daily returns
    st.write(f"Daily returns chart for {stock_symbol} from {start_date} to {end_date}")
    st.line_chart(df.set_index('Date')['daily_return'])

# Run the Streamlit app
if __name__ == '__main__':
    show()
