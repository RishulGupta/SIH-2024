import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.optimizers import Adam

# Paths to the dataset
colorized_path = "./dataset/colorised/"
greyscale_path = "./dataset/greyscale/"

# Create lists of file paths
colorized_files = [os.path.join(colorized_path, f"ROIs1868_summer_s2_59_p{i}.png") for i in range(1, 201)]
greyscale_files = [os.path.join(greyscale_path, f"ROIs1868_summer_s2_59_p{i}_greyscale.png") for i in range(1, 201)]

# Function to check if files exist and print missing ones
def check_files(colorized_files, greyscale_files):
    for color_path, gray_path in zip(colorized_files, greyscale_files):
        if not os.path.isfile(color_path):
            print(f"Color image file missing: {color_path}")
        if not os.path.isfile(gray_path):
            print(f"Grayscale image file missing: {gray_path}")

# Call the function to check for missing files
check_files(colorized_files, greyscale_files)

# Function to load and preprocess images
def load_data(colorized_files, greyscale_files, size=(128, 128)):
    X = []
    Y = []
    for color_path, gray_path in zip(colorized_files, greyscale_files):
        if not os.path.isfile(color_path) or not os.path.isfile(gray_path):
            print(f"Skipping missing files: {color_path} or {gray_path}")
            continue
        
        color_img = cv2.imread(color_path)
        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        
        if color_img is None or gray_img is None:
            print(f"Error loading images: {color_path} or {gray_path}")
            continue
        
        color_img = cv2.resize(color_img, size)
        gray_img = cv2.resize(gray_img, size)
        
        gray_img = np.expand_dims(gray_img, axis=-1)  # Convert to (height, width, 1)
        
        X.append(gray_img / 255.0)  # Normalize
        Y.append(color_img / 255.0)  # Normalize

    return np.array(X), np.array(Y)

# Load data
X, Y = load_data(colorized_files, greyscale_files)

# Ensure the loaded data has the expected shape
print(f"Loaded data shapes: X={X.shape}, Y={Y.shape}")

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

# Split the data into training and validation sets
X_train, X_val = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
Y_train, Y_val = Y[:int(0.8*len(Y))], Y[int(0.8*len(Y)):]

# Build and compile the model
input_shape = (128, 128, 1)  # (height, width, channels)
model = build_colorization_model(input_shape)
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

# Train the model
try:
    history = model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_data=(X_val, Y_val))

    # Save model weights with correct filename
    weights_path = 'colorization_model_weights.weights.h5'
    model.save_weights(weights_path)
    print(f"Model weights saved successfully to {weights_path}.")

except Exception as e:
    print(f"Error during training: {e}")

# Example of using the model
def colorize_image(model, input_image_path, output_image_path):
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {input_image_path}")
        return
    
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=-1) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    colorized_image = model.predict(image)
    colorized_image = (colorized_image[0] * 255).astype('uint8')  # Remove batch dimension and convert to [0, 255]
    
    cv2.imwrite(output_image_path, colorized_image)

# Example usage
colorize_image(model, 'black-and-white-image.jpg', './dataset/colorized_output.png')
