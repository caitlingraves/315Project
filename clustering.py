import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from CSV file
file_path = "user_interactions_with_names_filtered.csv"
data = pd.read_csv(file_path)

# Create a user-item interaction matrix
user_item_matrix = pd.pivot_table(data, values='interaction_count', index='user_id', columns='product_name', fill_value=0)

# Normalize the user-item interaction matrix
scaler = StandardScaler()
normalized_matrix = scaler.fit_transform(user_item_matrix)

                    # Used to determine the number of cluster for our k-mean
                    
# Define the range of clusters to test
cluster_range = range(1, 15)  # Adjust the range as needed

# Calculate WCSS for a range of cluster numbers
wcss = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_matrix)
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, wcss, marker='o')
plt.title("Elbow Method: WCSS vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

# Save the elbow method plot as an image file
elbow_image_path = "elbow.png"  # Desired file name
plt.savefig(elbow_image_path)
print(f"Elbow method plot saved to {elbow_image_path}")
# Close the plot
plt.close()

                               ##
             
# Define the optimal number of clusters (based on your elbow method analysis)
optimal_num_clusters = 4  # Adjust as needed
# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
kmeans.fit(normalized_matrix)

# Add cluster labels to the user-item interaction matrix
user_item_matrix['cluster'] = kmeans.labels_
# Create a plot to visualize common items within each cluster
plt.figure(figsize=(36, 20))
# Choose the optimal number of clusters (you may adjust this based on the elbow method plot)
num_clusters = [1, 2, 3, 4]


# For each cluster, identify the top 5 common items and plot them
for cluster in num_clusters:
    # Filter data for the current cluster
    cluster_users = user_item_matrix[user_item_matrix['cluster'] == cluster]
    
    # Calculate the sum of interaction counts for each product in the cluster
    product_interactions_sum = cluster_users.drop(columns='cluster').sum()
    
    # Identify the top 5 most common items in the cluster
    common_items = product_interactions_sum.sort_values(ascending=False).head(5)
    
    # Check if there are common items to plot
    #if len(common_items) > 0:
        # Plot the common items in each cluster separately using different colors and labels
    sns.barplot(x=common_items.values, y=common_items.index, label=f"Cluster {cluster}", alpha=0.8, edgecolor="black", width=0.8)

# Add title and labels
plt.title("Top 5 Common Items in Each Cluster")
plt.xlabel("Interaction Count")
plt.ylabel("Product Name")
plt.legend(title="Clusters")
 
# Save the plot as an image file
image_path = "cluster.png"  # Desired file name
plt.savefig(image_path)
print(f"K-means results saved to {image_path}")
# Close the plot
plt.close()