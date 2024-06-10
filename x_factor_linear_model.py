import streamlit as st
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    num_components = st.sidebar.number_input("Number of PCA Components", min_value=1, max_value=10, value=8)
    return start_date, end_date, num_components

def get_data_from_db(start_date, end_date):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    query = f"""
    SELECT date, symbol, open, close FROM reporting.mtd_daily_stock_data
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def add_daily_return_column(data):
    data['daily_return'] = (data['open'] - data['close']) / data['close']

def calculate_cumulative_returns(data):
    data['cumulative_return'] = (1 + data['daily_return']).groupby(data['symbol']).cumprod() - 1

def run_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca, pca_data

def train_linear_regression(pca_data, cumulative_returns):
    model = LinearRegression()
    model.fit(pca_data, cumulative_returns)
    return model

def compare_returns(actual, predicted):
    comparison = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    st.dataframe(comparison)
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R²): {r2}")
    st.write(f"Standard Deviation of Actual Returns: {actual.std()}")
    st.write(f"Standard Deviation of Predicted Returns: {predicted.std()}")

def show():
    # Get user input
    start_date, end_date, num_components = get_input()

    # Fetch data from the database
    df = get_data_from_db(start_date, end_date)
    # st.dataframe(df)

    # Add daily return column
    add_daily_return_column(df)

    # Prepare data for PCA
    daily_returns = df.pivot(index='symbol', columns='date', values='daily_return').fillna(0)

    # Calculate cumulative returns
    st.write('## Daily return and Cumulative return values')
    calculate_cumulative_returns(df)
    # st.dataframe(df)

    # Prepare cumulative returns for PCA
    cumulative_returns_series = df.groupby('symbol')['cumulative_return'].last().dropna()
    valid_symbols = cumulative_returns_series.index

    st.dataframe(daily_returns)

    # Ensure alignment between PCA data and cumulative returns
    daily_returns = daily_returns.loc[valid_symbols]

    # Run PCA
    pca, pca_data = run_pca(daily_returns, num_components)

    # Train linear regression model
    cumulative_returns = cumulative_returns_series.loc[valid_symbols].values
    model = train_linear_regression(pca_data, cumulative_returns)

    # Predict cumulative returns using the model
    predicted_returns = model.predict(pca_data)

    # Compare actual and predicted returns
    compare_returns(cumulative_returns, predicted_returns)

    # Display the data and summary statistics
    st.write(f"Showing data and summary statistics from {start_date} to {end_date}")
    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(df)

    with col2:
        st.dataframe(df.describe())

    # Plotting PCA components
    st.write("PCA components chart")
    for i in range(num_components):
        st.line_chart(pd.DataFrame(pca_data[:, i], index=daily_returns.index, columns=[f'PCA Component {i+1}']))

if __name__ == "__main__":
    show()
