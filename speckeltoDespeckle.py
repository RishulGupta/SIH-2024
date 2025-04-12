import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import (
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
    denoise_bilateral
)
from skimage.restoration import estimate_sigma
from skimage.util import random_noise
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# Load and preprocess image
image_path = 'full dataset/archive/v_2/agri/s1/ROIs1868_summer_s1_59_p11.png'  # Update with your local path
original_image = img_as_float(io.imread(image_path, as_gray=True))
noisy_image = random_noise(original_image, mode='speckle', var=0.01)

# Estimate sigma for NLM
sigma_est = np.mean(estimate_sigma(noisy_image))

# Apply NLM and its combinations
denoised_image_nlm = denoise_nl_means(noisy_image, h=1.0 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11, channel_axis=None)

# Combinations with NLM
denoised_image_nlm_tv = denoise_tv_chambolle(denoise_nl_means(noisy_image, h=1.0 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11, channel_axis=None), weight=0.1)
denoised_image_nlm_wavelet = denoise_wavelet(denoise_nl_means(noisy_image, h=1.0 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11, channel_axis=None), method='BayesShrink', mode='soft')
denoised_image_nlm_bilateral = denoise_bilateral(denoise_nl_means(noisy_image, h=1.0 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11, channel_axis=None), sigma_color=0.05, sigma_spatial=15)

# Combinations of the above
denoised_image_nlm_tv_wavelet = denoise_wavelet(denoise_tv_chambolle(denoise_nl_means(noisy_image, h=1.0 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11, channel_axis=None), weight=0.1), method='BayesShrink', mode='soft')
denoised_image_nlm_tv_bilateral = denoise_bilateral(denoise_tv_chambolle(denoise_nl_means(noisy_image, h=1.0 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11, channel_axis=None), weight=0.1), sigma_color=0.05, sigma_spatial=15)


# Define the CNN model for despeckling
def build_despeckle_model(input_shape):
    inputs = Input(shape=input_shape)

    # First convolutional layer with more filters
    x = Conv2D(128, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks with more filters
    for _ in range(8):
        residual = Conv2D(128, (3, 3), padding='same')(x)
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(128, (3, 3), padding='same')(residual)
        residual = BatchNormalization()(residual)
        x = Add()([x, residual])

    # Last convolutional layer
    outputs = Conv2D(1, (3, 3), padding='same')(x)
    outputs = BatchNormalization()(outputs)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs, outputs)
    return model


# Load and preprocess the SAR images
def load_images(image_paths, size=(128, 128)):
    images = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Unable to load image at {path}")
            continue  # Skip this image and move to the next one
        image = cv2.resize(image, size)
        image = image / 255.0  # Normalize to [0, 1]
        images.append(image)
    return np.array(images)

# Main function
def main():
    # Example image paths
    # Replace with your own paths
    speckled_image_paths = ["full dataset/archive/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png","full dataset/archive/v_2/agri/s1/ROIs1868_summer_s1_59_p3.png"]
    clean_image_paths = ["greyscale/ROIs1868_summer_s2_59_p2_greyscale.png","greyscale/ROIs1868_summer_s2_59_p3_greyscale.png"]

    # Load and preprocess images
    X = load_images(speckled_image_paths).reshape(-1, 128, 128, 1)
    Y = load_images(clean_image_paths).reshape(-1, 128, 128, 1)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)#error here

    # Build and compile the model
    input_shape = (128, 128, 1)
    model = build_despeckle_model(input_shape)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

    # Train the model
    batch_size = 32
    epochs = 5
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val))

    # Predict on a new SAR image
    test_image_path = 'full dataset/archive/v_2/agri/s1/ROIs1868_summer_s1_59_p2.png'  # Replace with your test image path
    test_image = load_images([test_image_path]).reshape(1, 128, 128, 1)
    despeckled_image = model.predict(test_image)

    # Save the despeckled image
    despeckled_image = despeckled_image.reshape(128, 128)
    despeckled_image = (despeckled_image * 255).astype('uint8')
    
    denoised_image_nlm_tv_wavelet_bilateral = denoise_bilateral(denoise_wavelet(denoise_tv_chambolle(denoise_nl_means(noisy_image, h=1.0 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11, channel_axis=None), weight=0.1), method='BayesShrink', mode='soft'), sigma_color=0.05, sigma_spatial=15)

    denoised_image_nlm_tv_wavelet_bilateral_uint8 = img_as_ubyte(denoised_image_nlm_tv_wavelet_bilateral)
    output_path = 'despeckled_image7.png'
    io.imsave(output_path, denoised_image_nlm_tv_wavelet_bilateral_uint8)

if __name__ == '__main__':
    main()

