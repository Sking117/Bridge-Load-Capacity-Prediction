import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Load the trained model and preprocessing pipeline
try:
    model = tf.keras.models.load_model('bridge_load_model.h5')
    preprocessor = joblib.load('preprocessing_pipeline.pkl')
except FileNotFoundError:
    st.error("Model or preprocessing pipeline files not found. Please ensure 'bridge_load_model.h5' and 'preprocessing_pipeline.pkl' are in the same directory.")
    st.stop()  # Stop execution if files are missing

# Function to make predictions
def predict_load(input_data):
    try:
        input_df = pd.DataFrame([input_data])
        processed_input = preprocessor.transform(input_df)
        prediction = model.predict(processed_input)
        return prediction[0][0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Streamlit app
def main():
    st.title("Bridge Maximum Load Capacity Prediction")

    # Input fields
    span_ft = st.number_input("Span (ft)", min_value=0.0, value=100.0)
    deck_width_ft = st.number_input("Deck Width (ft)", min_value=0.0, value=30.0)
    age_years = st.number_input("Age (Years)", min_value=0, value=50)
    num_lanes = st.number_input("Number of Lanes", min_value=1, value=2)
    condition_rating = st.number_input("Condition Rating", min_value=0, max_value=9, value=7)
    material = st.selectbox("Material", ['STEEL', 'CONCRETE', 'WOOD', 'MASONRY'])

    # Prediction button
    if st.button("Predict Maximum Load"):
        input_data = {
            'Span_ft': span_ft,
            'Deck_Width_ft': deck_width_ft,
            'Age_Years': age_years,
            'Num_Lanes': num_lanes,
            'Condition_Rating': condition_rating,
            'Material': material
        }

        prediction = predict_load(input_data)

        if prediction is not None:
            st.success(f"Predicted Maximum Load: {prediction:.2f} tons")

if __name__ == "__main__":
    main()
