import os
import numpy as np
import pickle
import json
import cv2
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "../model/best_model.pkl"
SCALER_PATH = "../model/scaler.pkl"
MODEL_DETAILS_PATH = "../model/model_details.json"
FEATURE_NAMES = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_DETAILS_PATH):
    raise FileNotFoundError("Required model files are missing! Train the model first.")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(MODEL_DETAILS_PATH, "r") as details_file:
    model_details = json.load(details_file)

best_accuracy = model_details.get("accuracy", "N/A")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_features_from_image(image_path):
    """Extracts features from a breast scan image using OpenCV."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (50, 50))
    features = resized_image.flatten()[:30]
    return np.array(features).reshape(1, -1)

@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict cancer diagnosis based on JSON input and return confidence score."""
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model or Scaler not found!"}), 500

        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        confidence = model.predict_proba(features_scaled).max()
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        feature_values = dict(zip(FEATURE_NAMES, features.tolist()[0]))

        return jsonify({
            "diagnosis": diagnosis,
            "accuracy": best_accuracy,
            "confidence": round(confidence * 100, 2),
            "features": feature_values
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/upload", methods=["POST"])
def upload_scan():
    """Handle image file uploads for cancer prediction with confidence score."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        extracted_features = extract_features_from_image(file_path)
        features_scaled = scaler.transform(extracted_features)
        prediction = model.predict(features_scaled)
        confidence = model.predict_proba(features_scaled).max()
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"

        feature_values = dict(zip(FEATURE_NAMES, extracted_features.tolist()[0]))

        return jsonify({
            "filename": filename,
            "diagnosis": diagnosis,
            "accuracy": best_accuracy,
            "confidence": round(confidence * 100, 2),
            "features": feature_values
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/uploads/<filename>", methods=["GET"])
def get_uploaded_file(filename):
    """Retrieve an uploaded file by filename."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Retrieve model details."""
    return jsonify(model_details)

if __name__ == "__main__":
    app.run(debug=True)