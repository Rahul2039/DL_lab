# -------------------------------
# Fix OpenMP Error (IMPORTANT)
# -------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------
# 1. Import Libraries
# -------------------------------
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# 2. Load Dataset
# -------------------------------
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# -------------------------------
# 3. Preprocessing (Padding)
# -------------------------------
max_len = 200

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# -------------------------------
# 4. Build LSTM Model
# -------------------------------
model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))

# Improved LSTM (for better marks)
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.summary()

# -------------------------------
# 5. Compile Model
# -------------------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -------------------------------
# 6. Train Model
# -------------------------------
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# -------------------------------
# 7. Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# -------------------------------
# 8. Plot Accuracy Graph
# -------------------------------
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

# -------------------------------
# 9. Plot Loss Graph
# -------------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()
