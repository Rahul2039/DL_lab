import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

# -------------------------------
# 1. Generate Sample Dataset
# -------------------------------
# (You can replace this with stock data CSV)

data = np.sin(np.arange(0, 100, 0.1))   # sine wave (time series)

# -------------------------------
# 2. Preprocessing
# -------------------------------
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1))

# create sequences
def create_dataset(dataset, time_steps=10):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i+time_steps])
        y.append(dataset[i+time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_dataset(data, time_steps)

# split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 3. Model Builder
# -------------------------------
def build_model(model_type):
    model = Sequential()
    
    if model_type == "RNN":
        model.add(SimpleRNN(32, input_shape=(time_steps, 1)))
    elif model_type == "LSTM":
        model.add(LSTM(32, input_shape=(time_steps, 1)))
    elif model_type == "GRU":
        model.add(GRU(32, input_shape=(time_steps, 1)))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    return model

# -------------------------------
# 4. Train and Evaluate Models
# -------------------------------
models = ["RNN", "LSTM", "GRU"]
results = {}

for m in models:
    print(f"\nTraining {m}...")
    
    model = build_model(m)
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )
    
    training_time = time.time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate loss (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    results[m] = {
        "Loss": mse,
        "Training Time": training_time
    }

# -------------------------------
# 5. Print Results
# -------------------------------
print("\n📊 Model Comparison:")
for model, res in results.items():
    print(f"{model}: Loss={res['Loss']:.6f}, Time={res['Training Time']:.2f}s")

# -------------------------------
# 6. Plot Predictions (Best Model)
# -------------------------------
best_model = min(results, key=lambda x: results[x]["Loss"])
print(f"\nBest Model: {best_model}")

model = build_model(best_model)
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

y_pred = model.predict(X_test)

plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title(f"{best_model} Prediction")
plt.legend()
plt.show()
