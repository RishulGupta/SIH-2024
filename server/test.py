import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = 'colorization_model.keras'
IMAGE_NAME = 'img1.png'
OUTPUT_NAME = 'img1.1.png'
SIZE = 256  # Assumes model expects 256x256

# Load model
model = tf.keras.models.load_model('./colorization_model.keras')

# Load and preprocess grayscale image
img = cv2.imread(IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Image '{IMAGE_NAME}' not found.")

img_resized = cv2.resize(img, (SIZE, SIZE))
img_normalized = img_resized.astype('float32') / 255.0
img_input = np.stack([img_normalized] * 3, axis=-1)  # replicate 1-channel to 3-channel
img_input = np.expand_dims(img_input, axis=0)

# Predict colorized image
prediction = model.predict(img_input)[0]
output_img = (prediction * 255).astype("uint8")

# Save output image
cv2.imwrite(OUTPUT_NAME, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
print(f"Colorized image saved as '{OUTPUT_NAME}'")
