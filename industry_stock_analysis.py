import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2024-05-01")
    end_date = st.sidebar.text_input("End Date", "2024-05-15")
    num_clusters = st.sidebar.number_input("Number of clusters", min_value=2, max_value=30, value=15)
    num_symbols_per_cluster = st.sidebar.number_input("Number of symbols to display per cluster", min_value=1, max_value=100, value=3)
    return start_date, end_date, num_clusters, num_symbols_per_cluster

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

def prepare_feature_matrix(stock_df):
    # Pivot the dataframe to have symbols as columns and dates as rows
    return_pivot = stock_df.pivot(index='Date', columns='Symbol', values='daily_return')
    
    # Fill missing values with 0
    return_pivot = return_pivot.fillna(0)
    
    # Standardize the data
    return_scaled = StandardScaler().fit_transform(return_pivot.T)
    
    return pd.DataFrame(return_scaled, index=return_pivot.columns)

def cluster_stocks(feature_matrix, num_clusters):
    # Perform PCA to reduce dimensionality
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(feature_matrix)
    
    # Perform clustering using KMeans on the PCA result
    model = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = model.fit_predict(pca_result)
    
    # Create a dataframe with symbols, PCA results, and their cluster labels
    cluster_df = pd.DataFrame({
        'symbol': feature_matrix.index,
        'x': pca_result[:, 0],
        'y': pca_result[:, 1],
        'cluster': clusters
    })
    
    return cluster_df, pca.explained_variance_ratio_

def monte_carlo_simulation(cluster_df, num_iterations=1000):
    cluster_counts = {cluster: 0 for cluster in cluster_df['cluster'].unique()}
    
    symbols = cluster_df['symbol'].tolist()
    for _ in range(num_iterations):
        selected_stock = np.random.choice(symbols)
        selected_cluster = cluster_df[cluster_df['symbol'] == selected_stock]['cluster'].values[0]
        cluster_counts[selected_cluster] += 1
    
    return cluster_counts

def get_top_symbols_per_cluster(cluster_df, stock_df, num_symbols_per_cluster):
    top_symbols_df_list = []
    for cluster in cluster_df['cluster'].unique():
        cluster_symbols = cluster_df[cluster_df['cluster'] == cluster]['symbol']
        cluster_volumes = stock_df[stock_df['Symbol'].isin(cluster_symbols)].groupby('Symbol')['Volume'].mean().reset_index()
        top_symbols = cluster_volumes.nlargest(num_symbols_per_cluster, 'Volume')['Symbol']
        top_symbols_df_list.append(cluster_df[cluster_df['symbol'].isin(top_symbols)])
    top_symbols_df = pd.concat(top_symbols_df_list)
    return top_symbols_df

def show():
    # Get user input
    start_date, end_date, num_clusters, num_symbols_per_cluster = get_input()

    # Fetch data from the CSV file
    stock_df = get_data_from_csv(start_date, end_date)

    # Calculate daily returns for each stock and index
    stock_df = calculate_daily_returns(stock_df)

    # Prepare the feature matrix
    feature_matrix = prepare_feature_matrix(stock_df)

    # Cluster stocks based on the feature matrix
    cluster_df, pca_e = cluster_stocks(feature_matrix, num_clusters)

    # Get top symbols per cluster based on volume
    top_symbols_df = get_top_symbols_per_cluster(cluster_df, stock_df, num_symbols_per_cluster)

    # Display the clustering result
    st.write("## Clustering Result")
    st.dataframe(top_symbols_df)

    # Plot the PCA result with clusters
    st.write("## PCA Result with Clusters")
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=top_symbols_df, x='x', y='y', hue='cluster', palette='viridis', s=100)
    for _, row in top_symbols_df.iterrows():
        plt.text(row['x'], row['y'], row['symbol'], horizontalalignment='left', size='medium', color='white', weight='semibold')
    plt.title('PCA Result with KMeans Clusters', color='white')
    plt.xlabel('PCA 1', color='white')
    plt.ylabel('PCA 2', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    st.pyplot(plt)


    # Identify under/over represented clusters
    st.write("## Average Values based on Clusters")
    stock_avg_df = stock_df.groupby(['Symbol']).mean()[['Close', 'Volume', 'daily_return']]
    cluster_df = cluster_df.join(stock_avg_df, 'symbol', 'left')

    cluster_avg_df = cluster_df[['x', 'y', 'cluster', 'Close', 'Volume', 'daily_return']].groupby(['cluster']).mean()[['x', 'y', 'Close', 'Volume', 'daily_return']]
    st.dataframe(cluster_avg_df)
    cluster_avg_df = cluster_avg_df.reset_index()

    # Sort the DataFrame by 'close' values
    cluster_avg_df = cluster_avg_df.sort_values(by='daily_return')

    # Create a bar chart using Altair
    chart = alt.Chart(cluster_avg_df).mark_bar().encode(
        x=alt.X('cluster:N', sort='-y'),
        y='daily_return:Q'
    ).properties(
        title='Clusters Ordered by Daily Return Values'
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

    # Identify under/over represented clusters
    median_return = cluster_avg_df['daily_return'].median()
    std_return = cluster_avg_df['daily_return'].std() * 1
    under_represented = cluster_avg_df[cluster_avg_df['daily_return'] < median_return - std_return]
    over_represented = cluster_avg_df[cluster_avg_df['daily_return'] > median_return + std_return]

    st.write("## Under-represented Clusters by Daily Returns")
    st.dataframe(under_represented)

    st.write("## Over-represented Clusters by Daily Returns")
    st.dataframe(over_represented)


    # Monte Carlo Simulation
    st.write("## Monte Carlo Simulation: Cluster Counts")
    cluster_counts = monte_carlo_simulation(cluster_df)
    
    # Convert the counts to a dataframe
    cluster_counts_df = pd.DataFrame.from_dict(cluster_counts, orient='index', columns=['count']).reset_index()
    cluster_counts_df.columns = ['cluster', 'count']
    
    # Order by count value
    cluster_counts_df = cluster_counts_df.sort_values(by='count', ascending=False)
    
    # Display the bar chart for cluster counts
    st.write("## Cluster Counts from Monte Carlo Simulation")
    chart = alt.Chart(cluster_counts_df).mark_bar().encode(
        x=alt.X('cluster:N', sort=None),
        y='count:Q'
    ).properties(
        title='Cluster Counts from Monte Carlo Simulation'
    )
    
    st.altair_chart(chart, use_container_width=True)

    # Identify under/over represented clusters
    mean_return = cluster_counts_df['count'].mean()
    std_return = cluster_counts_df['count'].std()

    under_represented = cluster_counts_df[cluster_counts_df['count'] < mean_return - std_return]
    over_represented = cluster_counts_df[cluster_counts_df['count'] > mean_return + std_return]

    st.write("## Under-represented Clusters by Count")
    st.dataframe(under_represented)

    st.write("## Over-represented Clusters by Count")
    st.dataframe(over_represented)

# Run the Streamlit app
if __name__ == '__main__':
    show()
