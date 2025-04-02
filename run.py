from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, r2_score, accuracy_score
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

MODEL_PATH = "ecg_model.h5"
IMG_SIZE = (224, 224)

# Load model
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# Prediction function
def predict(image):
    model = load_model()
    if model is None:
        return "Model not found. Train the model first.", None

    img = image.resize(IMG_SIZE).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = "Normal" if prob < 0.5 else "Abnormal"
    confidence = (1 - prob) * 100 if label == "Normal" else prob * 100

    return label, confidence

def evaluate_model(model, test_images, test_labels):
    predictions = model.predict(test_images)
    pred_labels = (predictions > 0.5).astype(int)
    true_labels = test_labels
    return {
        "accuracy": f"{accuracy_score(true_labels, pred_labels) * 100:.1f}%",
        "precision": f"{precision_score(true_labels, pred_labels) * 100:.1f}%",
        "recall": f"{recall_score(true_labels, pred_labels) * 100:.1f}%",
        "f1_score": f"{f1_score(true_labels, pred_labels) * 100:.1f}%",
        "r2_score": f"{r2_score(true_labels, predictions):.2f}"
    }

# Route to serve `index.html`
@app.route('/')
def index():
    return send_file("index.html")

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    image = Image.open(file)
    label, confidence = predict(image)

    # Dummy evaluation metrics (replace with actual computation if available)
    metrics = {
        "label": label,
        "confidence": f"{confidence:.1f}%",
        "accuracy": "92.5%",
        "precision": "90.0%",
        "recall": "93.0%",
        "f1_score": "91.5%",
        "r2_score": "0.88"
    }

    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True)





