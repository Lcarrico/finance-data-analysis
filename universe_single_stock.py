import streamlit as st
import pandas as pd
import psycopg2
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS
import matplotlib.pyplot as plt
import seaborn as sns



def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    symbol = st.sidebar.text_input("Symbol", "AAPL").upper()
    return start_date, end_date, symbol

def get_data_from_db(start_date, end_date, symbol, indices):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    query = f"""
    SELECT date, symbol, close FROM reporting.mtd_daily_stock_data
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
        AND symbol in ('{symbol}', {', '.join(['\''+i+'\'' for i in indices])})
    ORDER BY date, symbol
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def calculate_daily_returns(data):
    data['daily_return'] = data.groupby('symbol')['close'].pct_change()
    return data


def show():
    # Get user input
    start_date, end_date, symbol = get_input()

    # Fetch data from the database
    indices = ['QQQ', 'EEM', 'EFA']
    stock_df = get_data_from_db(start_date, end_date, symbol, indices)

    # Calculate daily returns for each stock and index
    stock_df = calculate_daily_returns(stock_df)

    # Display the data and summary statistics side by side
    st.write(f"Showing average daily returns for {symbol} and indices from {start_date} to {end_date}")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(stock_df)

    with col2:
        st.dataframe(stock_df.describe())

    
    # Plot the daily return values
    st.write("### Daily Returns Over Time")

    # Set the style for dark mode
    sns.set_theme(style="darkgrid")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, grp in stock_df.groupby(['symbol']):
        ax.plot(grp['date'], grp['daily_return'], label=key)
    
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
