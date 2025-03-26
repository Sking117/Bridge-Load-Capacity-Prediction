import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset (replace 'your_dataset.csv' with your actual file)
try:
    df = pd.read_csv('bridge_data.csv') # Assuming bridge_data.csv exists
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

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Model Development
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = ANN(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) #L2 regularization

# Training and Evaluation
epochs = 200
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10
trigger_times = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        val_losses.append(val_loss.item())

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'bridge_model.pth') #save best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print('Early stopping!')
            break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Load best model
model.load_state_dict(torch.load('bridge_model.pth'))

# Plot training/validation loss vs. epochs
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_mse = mean_squared_error(y_test.numpy(), test_outputs.numpy())
    print(f'Test MSE: {test_mse:.4f}')

# Save model files
torch.save(model.state_dict(), 'pytorch_bridge_model.pth')
print("Pytorch model saved as pytorch_bridge_model.pth")
