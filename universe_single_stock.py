import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    symbol = st.sidebar.text_input("Symbol", "AAPL").upper()
    return start_date, end_date, symbol

def get_data_from_csv(start_date, end_date, symbol):
    # Read the stock CSV file
    stock_df = pd.read_csv('sp500_stocks.csv')
    
    # Read the S&P 500 index CSV file
    index_df = pd.read_csv('sp500_index.csv')
    
    # Rename the 'S&P500' column to 'Close'
    index_df.rename(columns={'S&P500': 'Close'}, inplace=True)
    
    # Convert the 'Date' column to datetime format for both dataframes
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    index_df['Date'] = pd.to_datetime(index_df['Date'])
    
    # Filter the dataframes based on the date range and symbol
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    stock_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date) & (stock_df['Symbol'] == symbol)]
    index_df = index_df[(index_df['Date'] >= start_date) & (index_df['Date'] <= end_date)]
    
    return stock_df, index_df

def calculate_daily_returns(data, column='Close'):
    data['daily_return'] = data[column].pct_change()
    return data

def show():
    # Get user input
    start_date, end_date, symbol = get_input()

    # Fetch data from the CSV files
    stock_df, index_df = get_data_from_csv(start_date, end_date, symbol)

    # Calculate daily returns for the stock and the S&P 500 index
    stock_df = calculate_daily_returns(stock_df)
    index_df = calculate_daily_returns(index_df, 'Close')  # Assuming 'Close' is the column name in the index data

    # Merge the dataframes on the Date column
    merged_df = pd.merge(stock_df, index_df, on='Date', suffixes=('_stock', '_index'))

    # Display the data and summary statistics side by side
    st.write(f"Showing average daily returns for {symbol} and the S&P 500 index from {start_date} to {end_date}")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(merged_df)

    with col2:
        st.dataframe(merged_df.describe())

    # Plot the daily return values
    st.write("### Daily Returns Over Time")

    # Set the style for dark mode
    sns.set_theme(style="darkgrid")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(merged_df['Date'], merged_df['daily_return_stock'], label=symbol)
    ax.plot(merged_df['Date'], merged_df['daily_return_index'], label='S&P 500 Index')
    
    ax.legend(loc='best')
    ax.set_title('Daily Returns Over Time', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Daily Return', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    plt.xticks(rotation=45)
    fig.patch.set_facecolor('#0e1117')  # Background color for the figure
    ax.set_facecolor('#0e1117')  # Background color for the axes

    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    show()
