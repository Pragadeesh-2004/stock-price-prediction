import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
df = pd.read_excel("/content/NIFTY50_2013_to_2024_Merged_Sorted.xlsx", parse_dates=['Date'], index_col='Date')

# Use the entire dataset (2013-2024)
df = df[['Close']].dropna()  # Use only 'Close' price

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Define sequence length
SEQ_LENGTH = 365  # Use last 365 days to predict the next day

# Prepare X and y
X, y = [], []
for i in range(SEQ_LENGTH, len(df_scaled)):
    X.append(df_scaled[i - SEQ_LENGTH:i])  # Last 365 days as input
    y.append(df_scaled[i])  # Predict next day's 'Close' price

X, y = np.array(X), np.array(y)

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer with 1 neuron (only 'Close' price)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model on full dataset
print("Training model on the entire dataset (2013-2024)...")
model.fit(X, y, epochs=50, batch_size=32, verbose=1)  # Increased epochs for better accuracy

# Predict using the full dataset
y_pred = model.predict(X)

# Convert predictions back to actual values
y_actual = scaler.inverse_transform(y)
y_pred_actual = scaler.inverse_transform(y_pred)

# Error calculations
mse = mean_squared_error(y_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_actual, y_pred_actual)
r2 = r2_score(y_actual, y_pred_actual)
mape = np.mean(np.abs((y_actual - y_pred_actual) / y_actual)) * 100
accuracy = 100 - mape

# Print error metrics including accuracy
print(f"\n🔹 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 R-squared Score (R²): {r2:.4f}")
print(f"🔹 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"✅ Accuracy (100 - MAPE): {accuracy:.2f}%")

# Plot Actual vs Predicted Closing Prices
plt.figure(figsize=(12, 6))
plt.plot(y_actual, label="Actual Close Prices", color='blue')
plt.plot(y_pred_actual, label="Predicted Close Prices", color='red', linestyle='dashed')
plt.title("Actual vs Predicted Closing Prices")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# Function to predict future stock prices
def predict_future(days):
    last_seq = df_scaled[-SEQ_LENGTH:]  # Start with last 365 days
    predictions = []

    for _ in range(days):
        pred = model.predict(last_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
        predictions.append(pred[0])
        last_seq = np.append(last_seq[1:], pred).reshape(SEQ_LENGTH, 1)  # Update sequence

    return scaler.inverse_transform(np.array(predictions))

# Get predictions
next_1_day = predict_future(1)[0]
next_15_days = predict_future(15)
next_30_days = predict_future(30)

# Print predicted prices
print(f"\n📅 Predicted Close Price for Next Day: {next_1_day[0]:.2f}")
print(f"📅 Predicted Close Price for 15th Day: {next_15_days[14][0]:.2f}")
print(f"📅 Predicted Close Price for 30th Day: {next_30_days[29][0]:.2f}")

# Plot future trend for Close price
plt.figure(figsize=(12, 6))
plt.plot(range(1, 16), next_15_days, label="Next 15 Days Close Price", color='orange')
plt.plot(range(1, 31), next_30_days, label="Next 30 Days Close Price", color='green')
plt.axvline(x=15, linestyle='dashed', color='red', label="15th Day Prediction")
plt.axvline(x=30, linestyle='dashed', color='blue', label="30th Day Prediction")
plt.title("Future Closing Price Prediction")
plt.xlabel("Days Ahead")
plt.ylabel("Predicted Price")
plt.legend()
plt.show()
