import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
import urllib.request
import os




model_url = "https://raw.githubusercontent.com/Sking117/Bridge-Load-Capacity-Prediction/main/bridge_load_model.h5"
preprocessor_url = "https://raw.githubusercontent.com/Sking117/Bridge-Load-Capacity-Prediction/main/preprocessing_all.pkl"
model_path = "model_all.h5"
preprocessor_path = "preprocessing_all.pkl"

if not os.path.exists(model_path):
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        st.stop()

if not os.path.exists(preprocessor_path):
    try:
        urllib.request.urlretrieve(preprocessor_url, preprocessor_path)
    except Exception as e:
        st.error(f"Failed to download preprocessor: {e}")
        st.stop()

# ... rest of your Streamlit app code ...

try:
    model_all = keras.models.load_model(
        model_path, custom_objects={"mse": MeanSquaredError}
    )
    preprocessor_all = joblib.load(preprocessor_path)
except FileNotFoundError:
    st.error(
        "Model or preprocessing pipeline files not found. Please ensure 'model_all.h5' and 'preprocessing_all.pkl' are in the same directory or have been downloaded successfully."
    )
    st.stop()

# Function to make predictions
def predict_load(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        processed_input = preprocessor_all.transform(input_df)
        prediction = model_all.predict(processed_input)
        return prediction[0][0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Streamlit app
def main():
    st.title("Bridge Maximum Load Capacity Prediction (All Features)")

    # Input fields (adjust based on your actual features)
    span_ft = st.number_input("Span (ft)", min_value=0.0, value=100.0)
    deck_width_ft = st.number_input("Deck Width (ft)", min_value=0.0, value=30.0)
    age_years = st.number_input("Age (Years)", min_value=0, value=50)
    num_lanes = st.number_input("Number of Lanes", min_value=1, value=2)
    condition_rating = st.number_input("Condition Rating", min_value=0, max_value=9, value=7)
    material = st.selectbox("Material", ["STEEL", "CONCRETE", "WOOD", "MASONRY"])
    # Add other features based on what you included in 'preprocessor_all'
    # Example (adjust based on your actual data):
    # bridge_id = st.text_input("Bridge ID", "Bridge_1")
    # ... other features ...

    # Prediction button
    if st.button("Predict Maximum Load (All Features)"):
        input_data = {
            "Span_ft": span_ft,
            "Deck_Width_ft": deck_width_ft,
            "Age_Years": age_years,
            "Num_Lanes": num_lanes,
            "Condition_Rating": condition_rating,
            "Material": material,
            # Add other features here, matching the preprocessor_all
            # "Bridge_ID": bridge_id,
            # ... other features ...
        }

        prediction = predict_load(input_data)

        if prediction is not None:
            st.success(f"Predicted Maximum Load (All Features): {prediction:.2f} tons")

if __name__ == "__main__":
    main()
