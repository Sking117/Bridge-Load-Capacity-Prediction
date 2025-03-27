import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import pickle
import numpy as np

# Load the preprocessing pipeline
preprocessor_path = "preprocessor.pkl"
with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

# Transform training and test data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Define the ANN model
model = keras.Sequential([
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train_transformed.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="mse",
              metrics=["mae"])

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_transformed, y_train, 
                    validation_data=(X_test_transformed, y_test), 
                    epochs=100, 
                    batch_size=16, 
                    callbacks=[early_stopping], 
                    verbose=1)

# Save the trained model
model.save("bridge_load_ann.h5")

