import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# # Load data from files
# orders_df = pd.read_csv("orders.csv/orders.csv")
# order_products_prior_df = pd.read_csv("order_products__prior.csv/order_products__prior.csv")
# products_df = pd.read_csv("products.csv/products.csv")

# # Merge orders with order_products_prior to get user-item interactions
# order_product_merge = pd.merge(orders_df, order_products_prior_df, on='order_id', how='inner')

# # Filter out rows without the target user_id
# user_order_product_merge = order_product_merge[order_product_merge['user_id'].isin(range(1, 20001))]

# # Group by user_id and product_id to get interactions for each user and product
# user_item_interactions = user_order_product_merge.groupby(['user_id', 'product_id']).size().reset_index(name='interaction_count')

# # Filter interactions to include only interaction counts of 3 or above
# user_item_interactions_filtered = user_item_interactions[user_item_interactions['interaction_count'] >= 3]

# # Merge with products DataFrame to include product names
# user_item_interactions_with_names = pd.merge(user_item_interactions_filtered, products_df[['product_id', 'product_name']], on='product_id', how='left')

# # Save the simplified DataFrame to a new CSV file
# user_item_interactions_with_names.to_csv(f"user_interactions_with_names_filtered.csv", index=False)




#user_item_interactions_with_names = pd.read_csv("user_interactions_with_names_filtered.csv")


class ProductRecommendation:
    def __init__(self, data):
        self.data = data
        self.user_item_matrix = self.construct_user_item_matrix()
        self.product_names = data[['product_id', 'product_name']].drop_duplicates().set_index('product_id')
        self.product_ids = self.product_names.index.tolist()
        self.product_id_to_name = self.product_names.to_dict()['product_name']
        self.tf_idf_matrix = self.construct_tf_idf_matrix()

    def construct_user_item_matrix(self):
        return pd.pivot_table(self.data, values='interaction_count', index='user_id', columns='product_id', fill_value=0)

    def construct_tf_idf_matrix(self):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.product_names['product_name'])
        return tfidf_matrix

    def collaborative_filtering_recommendation(self, user_id, top_n=5):
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_interactions = self.user_item_matrix.loc[user_id]
        user_interactions_sorted = user_interactions.sort_values(ascending=False)
        interacted_products = user_interactions_sorted[user_interactions_sorted > 0].index.tolist()
        recommendations = []

        for product_id in interacted_products:
            similar_products = self.user_item_matrix.corrwith(self.user_item_matrix[product_id])
            similar_products = similar_products.dropna()
            similar_products = similar_products.sort_values(ascending=False)
            
            for idx in range(len(similar_products.index)):
                similar_product_id = similar_products.index[idx]
                correlation = similar_products.values[idx]
                if similar_product_id not in interacted_products and similar_product_id not in recommendations:
                    recommendations.append(similar_product_id)
                    if len(recommendations) == top_n:
                        return recommendations
        
        return recommendations

    def content_based_filtering_recommendation(self, user_id, top_n=5):
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_interactions = self.user_item_matrix.loc[user_id]
        interacted_products = user_interactions[user_interactions > 0].index.tolist()
        recommendations = []

        for product_id in interacted_products:
            idx = self.product_ids.index(product_id)
            cosine_similarities = linear_kernel(self.tf_idf_matrix[idx], self.tf_idf_matrix).flatten()
            related_products_indices = cosine_similarities.argsort()[::-1]
            related_products_ids = [self.product_ids[i] for i in related_products_indices]
            
            for related_product_id in related_products_ids:
                if related_product_id not in interacted_products and related_product_id not in recommendations:
                    recommendations.append(related_product_id)
                    if len(recommendations) == top_n:
                        return recommendations

        return recommendations

#Below is to get recommendations for 1 user:
# Read dataset from CSV
user_item_interactions_with_names = pd.read_csv("user_interactions_with_names_filtered.csv")

recommendation_system = ProductRecommendation(user_item_interactions_with_names)
user_id = 4337
top_n = 3

# Collaborative filtering recommendation
collab_filtering_recommendations = recommendation_system.collaborative_filtering_recommendation(user_id, top_n)
print(f"Collaborative Filtering Recommendations for user {user_id}:")
for product_id in collab_filtering_recommendations:
    print(f"Product ID: {product_id}, Product Name: {recommendation_system.product_id_to_name[product_id]}")

# Content-based filtering recommendation
content_based_recommendations = recommendation_system.content_based_filtering_recommendation(user_id, top_n)
print(f"\nContent-Based Filtering Recommendations for user {user_id}:")
for product_id in content_based_recommendations:
    print(f"Product ID: {product_id}, Product Name: {recommendation_system.product_id_to_name[product_id]}")


##############################################################################################33
#Below is to get recommendations for many users:
# user_item_interactions_with_names = pd.read_csv("user_interactions_with_names_filtered.csv")

# product_recommendation = ProductRecommendation(user_item_interactions_with_names)

# num_users = 6000
# recommendations_content = []
# recommendations_collab = []


# for user_id in range(1, num_users + 1):
#     user_recommendations_collab = product_recommendation.collaborative_filtering_recommendation(user_id, top_n=1)
#     user_recommendations_content = product_recommendation.content_based_filtering_recommendation(user_id, top_n=1)
    
#     if user_recommendations_collab:
#         recommendations_content.append((user_id, user_recommendations_content[0]))
#         recommendations_collab.append((user_id, user_recommendations_collab[0]))
#         print("user appended: ", user_id)

# products_df = pd.read_csv("products.csv/products.csv")
# products = pd.read_csv("products.csv/products.csv")

# recommendations_df = pd.DataFrame(recommendations_content, columns=['user_id', 'product_id'])
# r2 = pd.DataFrame(recommendations_collab, columns=['user_id', 'product_id'])

# r2 = pd.merge(r2, products[['product_id', 'product_name']], on='product_id', how='left')

# recommendations_df = pd.merge(recommendations_df, products[['product_id', 'product_name']], on='product_id', how='left')

# recommendations_df.to_csv('top_content.csv', index=False)
# r2.to_csv('top_collab.csv', index=False)
