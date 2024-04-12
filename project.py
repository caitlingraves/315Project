import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data from files
orders_df = pd.read_csv("orders.csv/orders.csv")
order_products_prior_df = pd.read_csv("order_products__prior.csv/order_products__prior.csv")
products_df = pd.read_csv("products.csv/products.csv")

# Select a single user (change the user_id as needed)

# Merge orders with order_products_prior to get user-item interactions
order_product_merge = pd.merge(orders_df, order_products_prior_df, on='order_id', how='inner')



# Filter out rows without the target user_id
user_order_product_merge = order_product_merge[order_product_merge['user_id'].isin(range(1, 301))]
# Group by product_id to get interactions for the target user
user_item_interactions = user_order_product_merge.groupby(['user_id', 'product_id']).size().reset_index(name='interaction_count')
# Merge with products DataFrame to include product names
user_item_interactions_with_names = pd.merge(user_item_interactions, products_df[['product_id', 'product_name']], on='product_id', how='left')

# Save the simplified DataFrame to a new CSV file
user_item_interactions_with_names.to_csv(f"user_interactions_with_names.csv", index=False)



# # Merge orders with order_products_prior to get user-item interactions
# order_product_merge = pd.merge(orders_df, order_products_prior_df, on='order_id', how='inner')

# # Filter out rows without a user_id
# order_product_merge = order_product_merge.dropna(subset=['user_id'])

# # Group by user_id and product_id to get user-item interactions and count occurrences
# user_item_interactions = order_product_merge.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)

# # Save the merged DataFrame to a new CSV file
# user_item_interactions.to_csv("user_item_interactions.csv")

# Compute user similarity matrix using cosine similarity
# user_similarity_matrix = cosine_similarity(user_item_interactions)

# # Function to generate recommendations for a target user
# def generate_recommendations(target_user_id, user_item_interactions, user_similarity_matrix, n_recommendations=3):
#     # Find the index of the target user in the user-item interactions matrix
#     target_user_index = user_item_interactions.index.get_loc(target_user_id)
    
#     # Get similarities of the target user with other users
#     target_user_similarities = user_similarity_matrix[target_user_index]

#     # Find users most similar to the target user
#     similar_users_indices = target_user_similarities.argsort()[::-1]  # Sort in descending order

#     # Initialize a list to store recommended item indices
#     recommended_items = []

#     # Iterate over similar users to find items to recommend
#     for user_index in similar_users_indices:
#         # Skip the target user
#         if user_index == target_user_index:
#             continue
        
#         # Find items that the similar user has ordered but the target user hasn't
#         unrated_items = user_item_interactions.iloc[user_index][user_item_interactions.iloc[target_user_index] == 0]
        
#         # Add unrated items to recommended items list
#         recommended_items.extend(unrated_items.index)
        
#         # Stop if enough recommendations are found
#         if len(recommended_items) >= n_recommendations:
#             break
    
#     return recommended_items[:n_recommendations]

# # Example usage
# target_user_id = 1  # Example target user ID
# recommendations = generate_recommendations(target_user_id, user_item_interactions, user_similarity_matrix)
# print("Recommendations for user", target_user_id, ":", recommendations)
