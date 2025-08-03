import streamlit as st
import pandas as pd
import pickle
import datetime as dt # Added back as it might be used internally in your full data processing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# --- Call set_page_config() FIRST ---
st.set_page_config(page_title="Shopper Spectrum", layout="wide", initial_sidebar_state="expanded")

# --- Cached Loaders - Now SAFE to call Streamlit functions inside IF an error occurs ---
@st.cache_resource
def load_kmeans_model(path='kmeans_model.pkl'):
    """Loads the trained K-Means model."""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        # Don't call st.error() here, just return None or raise an exception
        return None # We'll handle the error display outside this function

@st.cache_resource
def load_rfm_scaler(path='rfm_scaler.pkl'):
    """Loads the trained StandardScaler."""
    try:
        with open(path, 'rb') as file:
            scaler = pickle.load(file)
        return scaler
    except FileNotFoundError:
        return None # We'll handle the error display outside this function

@st.cache_data
def prepare_recommendation_data():
    """
    Prepares data for product recommendations: builds similarity matrix and product mapping.
    This uses a dummy dataset for demonstration. In a real app, load your actual df_cleaned.
    """
    # Dummy data, replace with your actual preprocessed df_cleaned
    df_cleaned = pd.DataFrame({
        'InvoiceNo': ['A', 'A', 'B', 'B', 'C', 'C', 'D'],
        'Description': ['BLUE VINTAGE SPOT BEAKER', 'PINK VINTAGE SPOT BEAKER', 'GREEN VINTAGE SPOT BEAKER', 'BLUE VINTAGE SPOT BEAKER', 'POTTING SHED CANDLE CITRONELLA', 'POTTING SHED ROSE CANDLE', 'PANTRY CHOPPING BOARD'],
        'StockCode': ['BVSB', 'PVSB', 'GVSB', 'BVSB', 'PSCC', 'PSRC', 'PCB'],
        'Quantity': [1, 1, 1, 1, 1, 1, 1],
        'CustomerID': [1, 1, 2, 2, 3, 3, 3]
    })
    # Create Customer-Product Matrix
    customer_product_matrix = df_cleaned.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)
    customer_product_matrix[customer_product_matrix > 0] = 1 # Binary matrix
    item_user_matrix = customer_product_matrix.T
    item_similarity_df = pd.DataFrame(cosine_similarity(item_user_matrix), index=item_user_matrix.index, columns=item_user_matrix.index)

    # Product Description Mapping
    stock_code_to_desc = df_cleaned.drop_duplicates('StockCode').set_index('StockCode')['Description'].to_dict()
    all_product_descriptions = sorted(df_cleaned['Description'].unique().tolist())
    return item_similarity_df, stock_code_to_desc, all_product_descriptions

# --- Load resources and handle errors immediately AFTER set_page_config() ---
kmeans_model = load_kmeans_model()
rfm_scaler = load_rfm_scaler()
item_similarity_df, stock_code_to_desc_map, all_product_names = prepare_recommendation_data()

# Display global error messages if models or data failed to load
if kmeans_model is None or rfm_scaler is None:
    st.error("Essential models (K-Means or Scaler) could not be loaded. Please ensure 'kmeans_model.pkl' and 'rfm_scaler.pkl' are in the script's directory.")
    st.stop() # Stop execution if critical resources are missing

if item_similarity_df is None:
    st.error("Recommendation data could not be prepared. Check your data loading/preprocessing logic in 'prepare_recommendation_data()'.")
    # Don't st.stop() here, as segmentation might still work

# --- Cluster Label Mapping (MUST match your notebook's final interpretation) ---
cluster_label_mapping = {
    0: 'High-Value Shopper',
    1: 'At-Risk/Occasional Shopper',
}

st.title("🛒 Shopper Spectrum Analytics")
st.markdown("---")

# Sidebar for Navigation
st.sidebar.title("Modules")
selection = st.sidebar.radio("", ["Product Recommendation", "Customer Segmentation"])

# --- Product Recommendation Module ---
if selection == "Product Recommendation":
    st.header("🎯 Product Recommendation")
    st.write("Find 5 similar products for an entered product name.")

    # Removed direct st.warning here as global checks are above
    # The logic inside this block will run only if item_similarity_df is not None

    product_input = st.selectbox("Enter Product Name:", all_product_names)

    if st.button("Get Recommendations"):
        if product_input:
            desc_to_stock_code = {v: k for k, v in stock_code_to_desc_map.items()}
            input_stock_code = desc_to_stock_code.get(product_input)

            if input_stock_code and input_stock_code in item_similarity_df.index:
                st.subheader(f"Recommended for '{product_input}':")
                similar_scores = item_similarity_df[input_stock_code].sort_values(ascending=False)
                similar_products = similar_scores[similar_scores.index != input_stock_code]
                top_5_recommendations_stock_codes = similar_products.head(5).index.tolist()

                if top_5_recommendations_stock_codes:
                    st.markdown("#### Top 5 Recommended Products:")
                    for i, stock_code in enumerate(top_5_recommendations_stock_codes):
                        rec_desc = stock_code_to_desc_map.get(stock_code, f"Product {stock_code} (Description N/A)")
                        st.markdown(f"**{i+1}. {rec_desc}**")
                else:
                    st.info("No similar products found for this item in the database.")
            else:
                st.warning(f"Product '{product_input}' not found or has no similarity data.")
        else:
            st.warning("Please select a product name.")

# --- Customer Segmentation Module ---
elif selection == "Customer Segmentation":
    st.header("🔍 Customer Segmentation")
    st.write("Predict customer segment based on Recency, Frequency, and Monetary values.")

    # Removed direct st.warning here as global checks are above
    # The logic inside this block will run only if kmeans_model and rfm_scaler are not None

    # 3 number inputs for RFM
    recency_input = st.number_input("Recency (days since last purchase):", min_value=1, value=300, step=1)
    frequency_input = st.number_input("Frequency (number of purchases):", min_value=1, value=5, step=1)
    monetary_input = st.number_input("Monetary (total spend):", min_value=0.01, value=150.0, format="%.2f")

    # Predict Cluster button
    if st.button("Predict Segment"):
        new_customer_rfm = pd.DataFrame([[recency_input, frequency_input, monetary_input]],
                                        columns=['Recency', 'Frequency', 'Monetary'])
        scaled_input = rfm_scaler.transform(new_customer_rfm)
        predicted_cluster_id = kmeans_model.predict(scaled_input)[0]
        predicted_segment_label = cluster_label_mapping.get(predicted_cluster_id, "Undefined Segment")
        st.success(f"**Predicted Customer Segment**: **{predicted_segment_label}**")

st.markdown("---")
st.caption("🛍️ Shopper Spectrum App | Collaborative Filtering & K-Means Clustering")