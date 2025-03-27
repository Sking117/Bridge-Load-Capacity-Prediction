import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
data = pd.read_csv(r"C:\Users\sydne\OneDrive\Documents\Computer Applications\lab_11_bridge_data.xlsx")

# Drop identifier column
data = data.drop(columns=['Bridge_ID'])

# Define features and target
selected_features = ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Material', 'Condition_Rating']
target = 'Max_Load_Tons'
X = data[selected_features]
y = data[target]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Condition_Rating']),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Material'])
    ])

X_processed = preprocessor.fit_transform(X)

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')

# Define model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                       kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32, callbacks=[early_stop], verbose=0)

# Save model
model.save('bridge_load_model.h5')

# Evaluate model
y_pred = model.predict(X_val).flatten()
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
print(f"MAPE: {mape:.2f}%")
