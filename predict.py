import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("saved_model/mnist_digit_model.h5")

# Load your own image
img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img / 255.0
img = np.expand_dims(img, axis=0)  # shape (1, 28, 28)

prediction = model.predict(img)
digit = np.argmax(prediction)

print("Predicted Digit:", digit)
