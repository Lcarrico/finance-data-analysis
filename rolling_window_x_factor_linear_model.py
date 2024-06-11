import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    num_components = st.sidebar.number_input("Number of PCA Components", min_value=1, max_value=5, value=3)
    rolling_window_count = st.sidebar.number_input("Rolling Window Count", min_value=6, max_value=10, value=6)
    return start_date, end_date, num_components, rolling_window_count

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

def add_daily_return_column(data):
    data['daily_return'] = (data['Open'] - data['Close']) / data['Close']

def calculate_cumulative_returns(data):
    data['cumulative_return'] = (1 + data['daily_return']).groupby(data['Symbol']).cumprod() - 1

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
                'Symbol': window['Symbol'].iloc[0],
                'Date': window['Date'].iloc[-1]
            }
            for i in range(len(window['Closing_Value'])):
                to_append[f'day_{i+1}'] = window['Closing_Value'].iloc[i]

            windows.append(to_append)

        return pd.DataFrame(windows)
    return create_rolling_windows

def show():
    # Get user input
    start_date, end_date, num_components, rolling_window_count = get_input()

    # Fetch data from the CSV file
    df = get_data_from_csv(start_date, end_date)

    # Add daily return column
    add_daily_return_column(df)

    # Prepare data for PCA
    daily_returns = df.pivot(index='Symbol', columns='Date', values='daily_return').fillna(0)

    # Calculate cumulative returns
    st.write('## Daily return and Cumulative return values')
    calculate_cumulative_returns(df)

    # Prepare cumulative returns for PCA
    cumulative_returns_series = df.groupby('Symbol')['cumulative_return'].last().dropna()
    valid_symbols = cumulative_returns_series.index

    # Ensure alignment between PCA data and cumulative returns
    daily_returns = daily_returns.loc[valid_symbols]
    
    # Melt the dataframe
    melted_df = daily_returns.reset_index().melt(id_vars=['Symbol'], var_name='Date', value_name='Closing_Value')

    # Convert date to datetime
    melted_df['Date'] = pd.to_datetime(melted_df['Date'])

    # Sort by symbol and date
    melted_df = melted_df.sort_values(by=['Symbol', 'Date'])

    # Apply the function to each group
    result = melted_df.groupby('Symbol').apply(create_rolling_windows_gen(rolling_window_count)).reset_index(drop=True)
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
