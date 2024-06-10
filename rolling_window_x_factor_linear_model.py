import streamlit as st
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from config import DB_HOST, DB_NAME, DB_USER, DB_PASS

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    num_components = st.sidebar.number_input("Number of PCA Components", min_value=1, max_value=5, value=3)
    rolling_window_count = st.sidebar.number_input("Rolling Window Count", min_value=6, max_value=10, value=6)
    return start_date, end_date, num_components, rolling_window_count

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

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def compare_returns(actual, predicted):
    comparison = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    st.write("## Model results")
    st.dataframe(comparison)
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R²): {r2}")
    st.write(f"Standard Deviation of Actual Returns: {actual.std()}")
    st.write(f"Standard Deviation of Predicted Returns: {predicted.std()}")


# Function to create rolling windows
def create_rolling_windows_gen(window_size):
    def create_rolling_windows(group):
        windows = []
        for i in range(len(group) - window_size + 1):
            window = group.iloc[i:i + window_size]
            to_append = {
                'symbol': window['symbol'].iloc[0],
                'date': window['date'].iloc[-1]
                # ,  # Take the last date in the window
                # 'day_1': window['closing_value'].iloc[0],
                # 'day_2': window['closing_value'].iloc[1],
                # 'day_3': window['closing_value'].iloc[2]
            }
            for i in range(len(window['closing_value'])):
                to_append[f'day_{i+1}'] = window['closing_value'].iloc[i]

            windows.append(to_append)

        return pd.DataFrame(windows)
    return create_rolling_windows


def show():
    # Get user input
    start_date, end_date, num_components, rolling_window_count = get_input()

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

    # Ensure alignment between PCA data and cumulative returns
    daily_returns = daily_returns.loc[valid_symbols]
    
    # Melt the dataframe
    melted_df = daily_returns.reset_index().melt(id_vars=['symbol'], var_name='date', value_name='closing_value')

    # Convert date to datetime
    melted_df['date'] = pd.to_datetime(melted_df['date'])

    # Sort by symbol and date
    melted_df = melted_df.sort_values(by=['symbol', 'date'])

    # Apply the function to each group
    result = melted_df.groupby('symbol').apply(create_rolling_windows_gen(rolling_window_count)).reset_index(drop=True)
    st.dataframe(result)
    
    # Run PCA
    X = result[[f'day_{i}' for i in range(1, rolling_window_count)]]
    y = result[f'day_{rolling_window_count}']
    pca, pca_data = run_pca(X, num_components)


    # Train linear regression model
    model = train_linear_regression(pca_data, y)

    # Predict cumulative returns using the model
    predicted_returns = model.predict(pca_data)

    # Compare actual and predicted returns
    compare_returns(y, predicted_returns)

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
        st.line_chart(pd.DataFrame(pca_data[:, i], index=result.index, columns=[f'PCA Component {i+1}']))

if __name__ == "__main__":
    show()
