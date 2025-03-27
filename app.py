import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load preprocessing pipeline and trained model
preprocessor_path = "preprocessor.pkl"
model_path = "bridge_load_ann.h5"

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

model = tf.keras.models.load_model(model_path)

# Streamlit App
st.title('Bridge Load Capacity Prediction')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')
def user_input_features():
    Span_ft = st.sidebar.slider('Span (ft)', 100, 1000, 300)
    Deck_Width_ft = st.sidebar.slider('Deck Width (ft)', 10, 50, 25)
    Age_Years = st.sidebar.slider('Age (years)', 1, 100, 50)
    Num_Lanes = st.sidebar.slider('Number of Lanes', 1, 8, 2)
    Condition_Rating = st.sidebar.slider('Condition Rating (1-5)', 1, 5, 3)
    Material = st.sidebar.selectbox('Material', ['Steel', 'Concrete', 'Composite'])
    
    data = {
        'Span_ft': Span_ft,
        'Deck_Width_ft': Deck_Width_ft,
        'Age_Years': Age_Years,
        'Num_Lanes': Num_Lanes,
        'Condition_Rating': Condition_Rating,
        'Material': Material
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Preprocess input data
input_transformed = preprocessor.transform(input_df)

# Predict the load capacity
prediction = model.predict(input_transformed)

# Display the prediction
st.subheader('Predicted Maximum Load Capacity (tons)')
st.write(f"{prediction[0][0]:.2f} tons")

