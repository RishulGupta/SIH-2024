import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image # type: ignore
import os
from dotenv import load_dotenv
import imghdr 

# load env variables
load_dotenv()

app = Flask(__name__)
model = tf.keras.models.load_model('models/colorization-GAN-with-autoencoders-agri.keras')
classify_model = tf.keras.models.load_model('models/classification.keras')

CORS_ORIGIN = os.environ.get('CORS_ORIGIN')
CORS(app, origins=[CORS_ORIGIN])

SIZE = 256
CLASS_NAMES = ['agriculture', 'barrenland', 'grassland', 'urban']

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def is_image(file):
    file_type = imghdr.what(file)
    return file_type in ['jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']

@app.route('/colorize', methods=['POST'])
def colorize():
    # validation
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not is_image(image):
        return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400

    # save image to disk
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], str(uuid.uuid4()) + '-' + image.filename)
    image.save(image_path)

    # preprocessing the image
    img = cv2.imread(image_path).astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    prediction = model.predict(img)

    # post processing the image
    output_img = (prediction[0] * 255).astype("uint8")

    # save processed image
    output_filename = os.path.join(app.config["OUTPUT_FOLDER"], "colorized_" + image.filename)
    cv2.imwrite(output_filename, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

    return send_file(output_filename, mimetype='image/png')

def preprocess_image(img):
    img = img.resize((256, 256))
    img = img.convert("RGB")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/classify', methods=['POST'])
def classify():
    # validation
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not is_image(image):
        return jsonify({"error": "Invalid file type. Only image files are allowed."}), 400
    
    img = Image.open(image)
    img_array = preprocess_image(img)

    predictions = classify_model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return jsonify({"class": predicted_class, "confidence": confidence})


if __name__ == '__main__':
    app.run(debug=True)
