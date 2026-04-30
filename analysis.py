import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors

def perform_analysis():
    stores = pd.read_csv('stores_data.csv')
    sales = pd.read_csv('sales_data.csv')
    
    # Aggregate demand data by store
    store_performance = sales.groupby('Store').agg({
        'Demand': ['mean', 'std', 'max'],
        'IsPromo': 'mean',
        'IsSeason': 'mean'
    }).reset_index()
    
    store_performance.columns = ['Store', 'AvgDemand', 'StdDemand', 'MaxDemand', 'PromoRatio', 'SeasonRatio']
    cluster_data = store_performance.merge(stores[['Store', 'SustainabilityRating']], on='Store')
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data.drop('Store', axis=1))
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # KNN for Similar Store Identification
    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(scaled_data)
    distances, indices = knn.kneighbors(scaled_data)
    # Store the 1st neighbor (itself) and 2nd neighbor (closest)
    cluster_data['Nearest_Store'] = indices[:, 1] + 1 # +1 because store IDs start at 1
    
    stores = stores.merge(cluster_data[['Store', 'Cluster', 'Nearest_Store']], on='Store')
    
    # Categorical Encoding
    le = LabelEncoder()
    categorical_cols = ['StoreType', 'Assortment', 'Region']
    for col in categorical_cols:
        stores[col + '_Encoded'] = le.fit_transform(stores[col])
        
    # PCA for Dimensionality Reduction
    pca_cols = ['SustainabilityRating', 'Cluster', 'StoreType_Encoded', 'Assortment_Encoded', 'Region_Encoded']
    pca_input = stores[pca_cols]
    pca_input_scaled = scaler.fit_transform(pca_input)
    
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(pca_input_scaled)
    
    stores['PCA1'] = pca_results[:, 0]
    stores['PCA2'] = pca_results[:, 1]
    
    stores.to_csv('stores_processed.csv', index=False)
    
    # Save visualizations
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=stores, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
    plt.title('Fashion Boutique Clusters (PCA Projection)')
    plt.savefig('store_clusters_pca.png')
    
    print("Analysis Complete: Clusters formed and PCA projection saved.")
    return stores

if __name__ == "__main__":
    perform_analysis()
