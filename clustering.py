import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('agg') # for matplotlib use

def perform_kmeans_clustering(file, num_clusters = 5):
    data = pd.read_csv(file)  # Load data from the CSV file
    
    features = data[['user_id', 'product_id', 'interaction_count']]  # Select the desired features to cluster
    
    # Normalization of the  the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42)
    data['cluster'] = kmeans.fit_predict(normalized_features)
    
    # Analyzation od the the results
    print(f"Cluster centers for {file}:")
    print(kmeans.cluster_centers_)
    
    # Visualization od the clusters
    plt.figure(figsize=(14, 10))
    
    #sns.scatterplot(x='user_id', y='product_id', hue='cluster', data=data, palette='viridis')
    sns.scatterplot(
       x='user_id', 
       y='product_id', 
       hue='cluster', 
       data=data, 
       palette='viridis',
       s=300, # Adjust to make the distance between points larger
       alpha=0.6  # Adjust the transparency of the points
    )
    
    plt.title(f"K-means Clustering of {file}")
    plt.xlabel("User ID")
    plt.ylabel("Product ID")
    
    plt.savefig(f"{file}_clustering.png") # Saving the imaages of the plots

# Perform K-means clustering separately for each CSV file
recommendations_filtered = "user_interactions_with_names_filtered.csv"
recommendations = "user_interactions_with_names.csv"

perform_kmeans_clustering(recommendations_filtered, num_clusters=5) 
perform_kmeans_clustering(recommendations, num_clusters=5)