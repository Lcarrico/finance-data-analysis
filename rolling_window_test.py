import pandas as pd

# Sample dataframe
data = {
    'symbol': ['AAPL', 'GOOGL'],
    '2024-05-20': [150.0, 2800.0],
    '2024-05-21': [152.0, 2820.0],
    '2024-05-22': [153.0, 2810.0],
    '2024-05-23': [154.0, 2830.0],
    '2024-05-24': [155.0, 2840.0]
}
df = pd.DataFrame(data)

# Melt the dataframe
melted_df = df.melt(id_vars=['symbol'], var_name='date', value_name='closing_value')

# Convert date to datetime
melted_df['date'] = pd.to_datetime(melted_df['date'])

# Sort by symbol and date
melted_df = melted_df.sort_values(by=['symbol', 'date'])

# Function to create rolling windows
def create_rolling_windows(group, window_size=3):
    windows = []
    for i in range(len(group) - window_size + 1):
        window = group.iloc[i:i + window_size]
        windows.append({
            'symbol': window['symbol'].iloc[0],
            'date': window['date'].iloc[-1],  # Take the last date in the window
            'day_1': window['closing_value'].iloc[0],
            'day_2': window['closing_value'].iloc[1],
            'day_3': window['closing_value'].iloc[2]
        })
    return pd.DataFrame(windows)

# Apply the function to each group
result = melted_df.groupby('symbol').apply(create_rolling_windows).reset_index(drop=True)

print(result)
