import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Housing Data Portfolio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Data and Model ---
@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    # Convert binary categorical features (yes/no) to numeric (1/0)
    binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})
    
    # Create numerical encoding for furnishingstatus for model input
    furnishing_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    df["furnishingstatus_encoded"] = df["furnishingstatus"].map(furnishing_map)
    return df

@st.cache_resource
def load_model_and_params():
    model_path = "best_model.joblib"
    params_path = "model_params.json"
    
    if not os.path.exists(model_path) or not os.path.exists(params_path):
        st.error("Model or parameters file not found. Please ensure the training script has run successfully.")
        return None, None
        
    model = joblib.load(model_path)
    with open(params_path, "r") as f:
        params = json.load(f)
    return model, params

df = load_data()
model, params = load_model_and_params()

if df is None or model is None or params is None:
    st.stop()

# --- Sidebar --- 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview & Visualizations", "Price Predictor"])
st.sidebar.markdown("--- ")
st.sidebar.info("This app analyzes housing data and provides price predictions.")

# --- Main App --- 
st.title("🏠 Housing Data Insights & Predictor")

if page == "Overview & Visualizations":
    st.header("Data Overview and Visualizations")
    
    st.markdown("This section provides an overview of the housing dataset and visualizes key relationships between features and price.")
    
    # Display basic info
    if st.checkbox("Show Raw Data Sample"):
        st.dataframe(df.head())
    
    st.markdown("--- ")
    
    # --- Visualizations ---
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Price Distribution
        st.subheader("Housing Price Distribution")
        fig_hist = px.histogram(df, x="price", nbins=20, title="Distribution of House Prices")
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Most houses are priced between 3M and 6M.")

    with col2:
        # 2. Price vs. Area
        st.subheader("Price vs. Area")
        fig_scatter = px.scatter(df, x="area", y="price", title="House Price vs. Area (sqft)", 
                                 labels={"area": "Area (sqft)", "price": "Price"},
                                 hover_data=["bedrooms", "bathrooms"])
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("There is a positive correlation between area and price.")

    st.markdown("--- ")

    # 3. Impact of Categorical Features
    st.subheader("Impact of Features on Average Price")
    features_categorical = [
        ("mainroad", "Main Road Access"),
        ("guestroom", "Guest Room"),
        ("basement", "Basement"),
        ("hotwaterheating", "Hot Water Heating"),
        ("airconditioning", "Air Conditioning"),
        ("prefarea", "Preferred Area"),
    ]
    selected_feature_id = st.selectbox("Select Feature:", options=[f[0] for f in features_categorical], 
                                       format_func=lambda x: dict(features_categorical)[x])
    
    avg_prices = df.groupby(selected_feature_id)["price"].mean().reset_index()
    avg_prices[selected_feature_id] = avg_prices[selected_feature_id].map({1: f"With {dict(features_categorical)[selected_feature_id]}", 0: f"Without {dict(features_categorical)[selected_feature_id]}"})
    
    fig_bar = px.bar(avg_prices, x=selected_feature_id, y="price", title=f"Average Price by {dict(features_categorical)[selected_feature_id]}",
                     labels={"price": "Average Price", selected_feature_id: ""},
                     color=selected_feature_id,
                     color_discrete_map={
                         f"With {dict(features_categorical)[selected_feature_id]}": "#3B82F6",
                         f"Without {dict(features_categorical)[selected_feature_id]}": "#F59E0B"
                     })
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(f"Houses with {dict(features_categorical)[selected_feature_id]} tend to have higher average prices.")

    st.markdown("--- ")

    # 4. Correlation Heatmap
    st.subheader("Feature Correlation Matrix")
    numeric_df = df[params["feature_names"] + ["price"]] # Use features from model params + price
    corr = numeric_df.corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
                       z=corr.values,
                       x=corr.columns,
                       y=corr.index,
                       hoverongaps = False,
                       colorscale="RdBu", # Red-Blue scale
                       zmin=-1, zmax=1))
    fig_heatmap.update_layout(title="Correlation Heatmap of Numerical Features")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.caption("Heatmap showing the correlation between different numerical features. Red indicates positive correlation, Blue indicates negative.")

elif page == "Price Predictor":
    st.header("Interactive House Price Predictor")
    st.markdown("Use the controls below to set property features and predict the house price based on the trained Linear Regression model.")

    # --- Prediction Input Form ---
    col1, col2 = st.columns([2, 1]) # Input form on left, prediction on right

    with col1:
        st.subheader("Property Features")
        
        # Get min/max from data for sliders/inputs
        min_area, max_area = int(df["area"].min()), int(df["area"].max())
        min_bed, max_bed = int(df["bedrooms"].min()), int(df["bedrooms"].max())
        min_bath, max_bath = int(df["bathrooms"].min()), int(df["bathrooms"].max())
        min_story, max_story = int(df["stories"].min()), int(df["stories"].max())
        min_park, max_park = int(df["parking"].min()), int(df["parking"].max())
        
        # Use sliders and number inputs for features used in the model
        area = st.slider("Area (sqft)", min_area, max_area, int(df["area"].median()))
        bedrooms = st.number_input("Bedrooms", min_bed, max_bed, int(df["bedrooms"].median()))
        bathrooms = st.number_input("Bathrooms", min_bath, max_bath, int(df["bathrooms"].median()))
        stories = st.number_input("Stories", min_story, max_story, int(df["stories"].median()))
        parking = st.number_input("Parking Spaces", min_park, max_park, int(df["parking"].median()))
        furnishingstatus = st.selectbox("Furnishing Status", options=["furnished", "semi-furnished", "unfurnished"], index=1)
        
        # Add other features as checkboxes (even if not used by this specific model, for UI consistency)
        st.markdown("**Other Features:**")
        c1, c2 = st.columns(2)
        with c1:
            mainroad = st.checkbox("Main Road Access", value=True)
            basement = st.checkbox("Basement", value=False)
            airconditioning = st.checkbox("Air Conditioning", value=True)
        with c2:
            guestroom = st.checkbox("Guest Room", value=False)
            hotwaterheating = st.checkbox("Hot Water Heating", value=False)
            prefarea = st.checkbox("Preferred Area", value=False)

    # --- Prediction Logic ---
    # Map furnishing status to encoded value
    furnishing_encoded = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}[furnishingstatus]
    
    # Create feature array in the correct order based on model_params
    input_features = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "furnishingstatus_encoded": furnishing_encoded
    }])
    
    # Ensure the order matches the training order
    input_features = input_features[params["feature_names"]]
    
    # Scale the features using loaded parameters
    # Note: The model pipeline already includes the scaler
    predicted_price = model.predict(input_features)[0]
    
    # Ensure prediction is not negative
    predicted_price = max(0, predicted_price)

    with col2:
        st.subheader("Prediction Result")
        st.metric(label="Predicted House Price", value=f"${predicted_price:,.0f}")
        st.caption(f"Based on the selected features. Model: {params.get('model_name', 'Linear Regression')}")
        
        st.markdown("--- ")
        
        # Display Feature Importance (Coefficients for Linear Regression)
        st.subheader("Feature Importance (Model Coefficients)")
        importance_df = pd.DataFrame({
            "feature": params["feature_names"],
            "importance": np.abs(params["coefficients"]) # Absolute value of coefficients
        }).sort_values("importance", ascending=False)
        
        # Make feature names more readable
        name_map = {
            "area": "Area",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "stories": "Stories",
            "parking": "Parking",
            "furnishingstatus_encoded": "Furnishing"
        }
        importance_df["feature_readable"] = importance_df["feature"].map(name_map)
        
        fig_importance = px.bar(importance_df, x="importance", y="feature_readable", orientation="h",
                               title="Impact of Features on Prediction (Absolute Coefficient)")
        fig_importance.update_layout(yaxis_title="Feature", xaxis_title="Absolute Coefficient Value")
        st.plotly_chart(fig_importance, use_container_width=True)
        st.caption("Shows the magnitude of each feature's coefficient in the linear model.")
