import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.title("Digit Recognition App 🧠🔢")

uploaded_file = st.file_uploader("Upload a 28x28 digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    st.image(img, caption="Uploaded Image", width=150)

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    model = tf.keras.models.load_model("mnist_digit_model.h5")
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {digit}")
