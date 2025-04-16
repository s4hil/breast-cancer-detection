import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from deepseek import analyze_results  # Custom DeepSeek diagnostic tool
import json 
import pickle
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model_path_keras = os.path.join(os.path.dirname(__file__), 'breast_cancer_model.h5')
model_keras = load_model(model_path_keras)

with open("../model/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("../model/model_details.json", "r") as f:
    model_details = json.load(f)

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

@app.route("/")
def home():
    return render_template("index.html")

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(48, 48))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize
    return img

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return render_template("error.html", error_message="No file part in the request")

    file = request.files['file']
    if file.filename == '':
        return render_template("error.html", error_message="No selected file")

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_img = preprocess_image(filepath)
        result = model_keras.predict(processed_img)[0][0]
        prediction = "Benign" if result > 0.5 else "Malignant"
        confidence = result * 100 if prediction == "Benign" else (1 - result) * 100

        # Use DeepSeek for further diagnosis support
        deepseek_output = analyze_results(prediction, round(confidence, 2), accuracy=98.4)  # update accuracy if needed

        return render_template(
            "result.html",
            file_path=os.path.join("uploads", filename).replace("\\", "/"),
            diagnosis=prediction,
            confidence=round(confidence, 2),
            accuracy=98.4,
            deepseek=deepseek_output
        )
    except Exception as e:
        return render_template("error.html", error_message=str(e))

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/predict', methods=['POST'])
def predict_json():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        
        missing_features = [f for f in FEATURE_NAMES if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing features: {', '.join(missing_features)}",
                "required_features": FEATURE_NAMES
            }), 400
            
        invalid_features = {}
        for feature in FEATURE_NAMES:
            try:
                float(data[feature])
            except (ValueError, TypeError):
                invalid_features[feature] = data[feature]
                
        if invalid_features:
            return jsonify({
                "error": "Invalid feature values",
                "invalid_features": invalid_features
            }), 400
            
        features = [float(data[feature]) for feature in FEATURE_NAMES]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        confidence = model.predict_proba(features_scaled)[0].max() * 100
        diagnosis = "Malignant" if prediction == 1 else "Benign"
        
        deepseek_output = analyze_results(diagnosis, round(confidence, 2), round(model_details.get("accuracy", 0) * 100, 2))


        return jsonify({
            "diagnosis": diagnosis,
            "confidence": round(confidence, 2),
            "accuracy": round(model_details.get("accuracy", 0) * 100, 2),
            "features_used": {name: val for name, val in zip(FEATURE_NAMES, features)},
            "deepseek": deepseek_output
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
