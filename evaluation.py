import pandas as pd
from project import *

# Read data from CSV files
orders_df = pd.read_csv('orders.csv')
order_products_prior_df = pd.read_csv('order_products__prior.csv')
products_df = pd.read_csv('products.csv')

# Merge orders and order_products_prior on 'order_id'
order_user_products_df = pd.merge(orders_df, order_products_prior_df, on='order_id')

# Merge the result with products on 'product_id' to get product names
order_user_products_df = pd.merge(order_user_products_df, products_df, on='product_id')

# Group the merged data by user_id and create a dictionary associating each user with a set of actual products
user_actual_dict = order_user_products_df.groupby('user_id')['product_id'].apply(set).to_dict()

# Read the final result files
result_df_1 = pd.read_csv('user_interactions_with_names_filtered.csv')
result_df_2 = pd.read_csv('user_interactions_with_names.csv')

# Define the K values you want to evaluate
k_values = [5, 20, 50, 100]

# Function to calculate Precision@K and store the results in a dictionary
def calculate_precision_at_k(user_actual_dict, result_df, k_values):
    precision_results = {}
    for k in k_values:
        precision_list = []
        for user_id in result_df['user_id'].unique():
            # Get the top K recommended products for the user
            top_k_recommended = result_df.loc[result_df['user_id'] == user_id, 'product_id'].head(k).values
            # Get the actual products purchased by the user
            actual_products = user_actual_dict.get(user_id, set())
            # Calculate the number of relevant items in the top K recommendations
            relevant_recommended = len(set(top_k_recommended) & actual_products)
            # Calculate Precision@K for this user
            precision_at_k = relevant_recommended / k
            # Append the user's Precision@K to the list
            precision_list.append(precision_at_k)
        # Calculate the average Precision@K across all users
        average_precision_at_k = sum(precision_list) / len(precision_list)
        # Store the result
        precision_results[k] = average_precision_at_k
    return precision_results

# Calculate Precision@K for the first final result file
precision_at_k_1 = calculate_precision_at_k(user_actual_dict, result_df_1, k_values)
# Calculate Precision@K for the second final result file
precision_at_k_2 = calculate_precision_at_k(user_actual_dict, result_df_2, k_values)

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Result File': ['ResultFiltered', 'ResultPlain'],
    # Precision@K
    'Precision@5': [precision_at_k_1[5], precision_at_k_2[5]],
    'Precision@20': [precision_at_k_1[20], precision_at_k_2[20]],
    'Precision@50': [precision_at_k_1[50], precision_at_k_2[50]],
    'Precision@100': [precision_at_k_1[100], precision_at_k_2[100]]
})
# Save the results DataFrame to a CSV file
results_df.to_csv('results_accurary.csv', index=False, float_format='%.5f')