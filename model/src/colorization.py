
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from tqdm import tqdm
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import kagglehub


# -------------------- SORT FILES ALPHANUMERIC --------------------
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    return sorted(data, key=lambda key: [convert(c) for c in re.split('([0-9]+)', key)])

# -------------------- LOAD DATA --------------------
SIZE = 256
color_img, gray_img = [], []

# Download the dataset via kagglehub
# path = kagglehub.dataset_download("requiemonk/sentinel12-image-pairs-segregated-by-terrain")


# Set correct paths for SAR (grayscale) and RGB (color)
# color_path = os.path.join(path,'v_2', 'agri', 's2')
# gray_path = os.path.join(path,'v_2', 'agri', 's1')

color_path = "./data/colorization/color"
gray_path = "./data/colorization/gray"

# Read and preprocess color images
for file in tqdm(sorted_alphanumeric(os.listdir(color_path)), desc="Loading color images"):
    img = cv2.imread(os.path.join(color_path, file), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    color_img.append(img_to_array(img))

# Read and preprocess gray images
for file in tqdm(sorted_alphanumeric(os.listdir(gray_path)), desc="Loading grayscale images"):
    img = cv2.imread(os.path.join(gray_path, file), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    gray_img.append(img_to_array(img))

# Convert to numpy arrays
color_img = np.array(color_img)
gray_img = np.array(gray_img)

# Train-test split
# train_color = tf.data.Dataset.from_tensor_slices(color_img[:2000]).batch(16)
# train_gray = tf.data.Dataset.from_tensor_slices(gray_img[:2000]).batch(16)
# test_color = tf.data.Dataset.from_tensor_slices(color_img[2000:]).batch(8)
# test_gray = tf.data.Dataset.from_tensor_slices(gray_img[2000:]).batch(8)

split_index = int(0.8 * len(color_img))
train_color = tf.data.Dataset.from_tensor_slices(color_img[:split_index]).batch(16)
train_gray = tf.data.Dataset.from_tensor_slices(gray_img[:split_index]).batch(16)
test_color = tf.data.Dataset.from_tensor_slices(color_img[split_index:]).batch(8)
test_gray = tf.data.Dataset.from_tensor_slices(gray_img[split_index:]).batch(8)

# -------------------- MODEL DEFINITIONS --------------------
def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer='he_normal', use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer='he_normal', use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [downsample(f, 4, i != 0) for i, f in enumerate([64, 128, 256, 512, 512, 512, 512, 512])]
    up_stack = [upsample(512, 4, True) for _ in range(3)] + [upsample(512, 4)] + [upsample(f, 4) for f in [256, 128, 64]]
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
    x, skips = inputs, []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    inp = tf.keras.layers.Input(shape=[256, 256, 3])
    tar = tf.keras.layers.Input(shape=[256, 256, 3])
    x = tf.keras.layers.Concatenate()([inp, tar])
    x = downsample(64, 4, False)(x)
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer='he_normal', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer='he_normal')(x)
    return tf.keras.Model(inputs=[inp, tar], outputs=x)

# -------------------- TRAINING --------------------
generator = Generator()
discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
LAMBDA = 100
genLoss, discLoss = [], []

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    genLoss.append(total_gen_loss)
    return total_gen_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    discLoss.append(total_disc_loss)
    return total_disc_loss
def compute_enl(image):
    # Ensure tensor format
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Compute mean and standard deviation
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image)

    # Compute ENL (avoid division by zero)
    enl = (mean ** 2) / (std ** 2 + 1e-8)  # Adding small value to avoid division by zero

    return enl.numpy()
lpips_model = lpips.LPIPS(net='vgg')

def compute_lpips_torch(image1, image2):
    # Convert to Tensor and normalize to [-1, 1]
    image1 = torch.tensor(image1.numpy()).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    image2 = torch.tensor(image2.numpy()).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

    # Compute LPIPS
    lpips_distance = lpips_model(image1, image2)

    return lpips_distance.item()

# Compute LPIPS using PyTorch
lpips_value_torch = compute_lpips_torch(img_1, prediction)
print("LPIPS (PyTorch):", lpips_value_torch)
img_1_gray = tf.image.rgb_to_grayscale(tf.convert_to_tensor(img_1, dtype=tf.float32))
prediction_gray = tf.expand_dims(prediction[..., 0], axis=-1)
image_for_test_gray = tf.squeeze(image_for_test, axis=0)

enl_img_1 = compute_enl(img_1_gray)
enl_prediction = compute_enl(prediction_gray)
enl_test = compute_enl(image_for_test_gray)

print("ENL (img_1, Grayscale):", enl_img_1)
print("ENL (Prediction, First Channel):", enl_prediction)
print("ENL (Image for Test, Grayscale):", enl_test)
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

def fit(train_ds, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        for input_image, target in tf.data.Dataset.zip((train_gray, train_color)):
            train_step(input_image, target)
def mse(image_true, image_pred):
    return np.mean((image_true - image_pred) ** 2)
for example_input, example_target in tf.data.Dataset.zip((gray_dataset,color_dataset)).take(1):
    print(mse(example_input, example_target))
# Train for 10 epochs
fit(tf.data.Dataset.zip((train_gray, train_color)), epochs=10)
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 1.0  # If images are normalized [0,1]
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

# Compute PSNR for all images
psnr_values = [psnr(gt, pred) for gt, pred in zip(color_dataset_t, preds)]
mean_psnr = np.mean(psnr_values)

print(f"Mean PSNR: {mean_psnr:.2f} dB")
# Save the generator model
generator.save("colorization_model.keras")
def compute_mse(image1, image2):
    """
    Compute Mean Squared Error (MSE) between two images.
    
    :param image1: Ground truth image (NumPy array)
    :param image2: Predicted/generated image (NumPy array)
    :return: MSE value
    """
    return np.mean((image1 - image2) ** 2)

# Compute MSE for the entire dataset
mse_values = [compute_mse(gt, pred) for gt, pred in zip(ground_truth_images, preds)]

# Calculate mean MSE across all images
mean_mse = np.mean(mse_values)

print(f"Mean MSE: {mean_mse:.4f}")