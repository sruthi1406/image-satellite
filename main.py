import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Config
IMG_SIZE = 128
CLASSES = ['land', 'ocean', 'desert', 'cloud']
DATASET_PATH = 'C:/Users/Cyber/Desktop/imgsatellite'

# 1. Load data
def load_data(dataset_path, img_size=128):
    X, Y = [], []
    for idx, cls in enumerate(CLASSES):
        folder_path = os.path.join(dataset_path, cls)
        label = [0] * len(CLASSES)
        label[idx] = 1.0
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0
                X.append(img)
                Y.append(label)
            except:
                print(f"Error loading: {img_path}")
                continue
    return np.array(X), np.array(Y)

print("[INFO] Loading data...")
X, Y = load_data(DATASET_PATH, IMG_SIZE)

# 2. Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# 3. Build CNN Model
def build_composition_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(4, activation='softmax')  # Outputs percentages
    ])
    return model

print("[INFO] Building model...")
model = build_composition_model()
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train model
print("[INFO] Training model...")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=25,
    batch_size=32
)

# 5. Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 6. Evaluate on validation set
print("[INFO] Evaluating model...")
val_loss, val_accuracy = model.evaluate(X_val, Y_val)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
model.save("model.h5")
