import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_dataset.csv' with your actual file)
try:
    df = pd.read_csv(r"C:\Users\sydne\Downloads\lab_11_bridge_data.csv") # Assuming bridge_data.csv exists
except FileNotFoundError:
    print("Error: bridge_data.csv not found. Please ensure the file is in the correct location.")
    exit()

# Data Exploration and Preprocessing
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values (example: fill with mean for numerical, mode for categorical)
for column in df.columns:
    if df[column].isnull().any():
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

# Encode categorical variables (example: 'Bridge_Type')
le = LabelEncoder()
df['Bridge_Type'] = le.fit_transform(df['Bridge_Type'])

# Normalize/standardize features
X = df.drop('Max_Load_Tons', axis=1)
y = df['Max_Load_Tons']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Development (using sklearn's MLPRegressor)
model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                     alpha=1e-5, max_iter=500, early_stopping=True, validation_fraction=0.2,
                     n_iter_no_change=10, random_state=42)

# Training and Evaluation
model.fit(X_train, y_train)

# Plot training/validation loss vs. epochs (using loss_curve_)
plt.plot(model.loss_curve_, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation on test set
y_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {test_mse:.4f}')

#Saving the model.
import pickle
with open('sklearn_bridge_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("sklearn model saved as sklearn_bridge_model.pkl")
