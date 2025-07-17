import os
import shutil
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = 'C:/Users/Cyber/Desktop/imgsatellite/model.h5'  # Already trained model
INPUT_FOLDER = 'C:/Users/Cyber/Desktop/imagesatellited/input_images'  # Folder with raw/unfiltered images
OUTPUT_FOLDER = 'C:/Users/Cyber/Desktop/imgsatellite/filtered_images'
IMG_SIZE = 128
CLASSES = ['land', 'ocean', 'desert', 'cloud']

# Load trained model
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to preprocess image
def preprocess_image(image_path, img_size=128):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Function to predict image composition
def predict_composition(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img, verbose=0)[0]
    return dict(zip(CLASSES, prediction * 100))

# Process each image in input folder
print("[INFO] Processing images...")
for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(INPUT_FOLDER, file)
        try:
            composition = predict_composition(image_path)
            ocean_percentage = composition.get('ocean', 0)

            if ocean_percentage > 40:
                shutil.copy(image_path, os.path.join(OUTPUT_FOLDER, file))
                print(f"✔ Copied: {file} | Ocean: {ocean_percentage:.2f}%")
            else:
                print(f"✘ Skipped: {file} | Ocean: {ocean_percentage:.2f}%")

        except Exception as e:
            print(f"[ERROR] {file}: {e}")

print("[DONE] Ocean filtering complete.")
