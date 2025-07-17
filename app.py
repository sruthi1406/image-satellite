# -----------------------------
# SATELLITE IMAGE COMPOSITION STREAMLIT APP
# -----------------------------

# Imports
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Satellite Image Composition Predictor", layout="centered")

# Constants
IMG_SIZE = 128
CLASSES = ['forest', 'land', 'ocean','cloud']

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Title
st.title("🛰️ Satellite Image Composition Predictor")
st.markdown("""
Upload a satellite image and this model will estimate the composition of:
- 🌍 Land  
- 🌊 Ocean  
- 🏜️ Desert  
- ☁️ Clouds  
""")

# Upload
uploaded_file = st.file_uploader("📤 Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
    image_input = image_resized / 255.0

    # Predict
    prediction = model.predict(np.expand_dims(image_input, axis=0))[0]
    composition = {cls: round(percent * 100, 2) for cls, percent in zip(CLASSES, prediction)}

    # Show image
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    # Show results
    st.subheader("📊 Predicted Composition (%)")
    for cls, percent in composition.items():
        st.write(f"**{cls.capitalize()}**: {percent}%")

    # Show chart
    fig, ax = plt.subplots()
    ax.bar(composition.keys(), composition.values(), color=['green', 'peru','blue','pink'])
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.set_title("Predicted Image Composition")
    st.pyplot(fig)

else:
    st.info("Please upload a satellite image to get predictions.")
