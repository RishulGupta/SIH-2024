import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam

# Define U-Net architecture for colorization
def build_colorization_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c7)  # 3 channels for RGB color

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Load the pre-trained model weights
def load_pretrained_model(weights_path, input_shape):
    model = build_colorization_model(input_shape)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
    model.load_weights(weights_path)
    return model

# Function to colorize an image
def colorize_image(model, input_image_path, output_image_path):
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {input_image_path}")
        return
    
    # Resize and preprocess the image
    image = cv2.resize(image, (256, 256))  # Match the input size of the model
    image = np.expand_dims(image, axis=-1)  # Convert to (height, width, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize
    
    # Predict and save the output
    colorized_image = model.predict(image)
    colorized_image = (colorized_image[0] * 255).astype('uint8')  # Convert to [0, 255]
    
    cv2.imwrite(output_image_path, colorized_image)
    print(f"Colorized image saved to {output_image_path}")

# Paths to pre-trained model weights and image
weights_path = 'colorization_model_weights.weights.h5'
input_image_path = 'flower.jpg'
output_image_path = './dataset/colorized_output.png'

# Load the model and colorize the image
input_shape = (256, 256, 1)  # (height, width, channels)
model = load_pretrained_model(weights_path, input_shape)
colorize_image(model, input_image_path, output_image_path)
