from flask import Flask, render_template, request
import os
import base64
import cv2

from utils import (
    get_prediction,
    explain_image,
    is_valid_plant_image,
    generate_explanation,
    get_disease_details
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ LANDING PAGE
@app.route("/")
def landing():
    return render_template("landing.html")

# ✅ DASHBOARD
@app.route("/dashboard")
def dashboard():
    return render_template("index.html", active_section="upload")

# ✅ PREDICTION
@app.route("/explain", methods=["POST"])
def explain():
    file = request.files.get("image")

    if not file:
        return render_template("index.html",
                               label="No file uploaded",
                               confidence=0,
                               explanation_text="Please upload an image",
                               active_section="prediction")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    if not is_valid_plant_image(filepath):
        return render_template("index.html",
                               label="Invalid Image",
                               confidence=10,
                               explanation_text="Not a plant image",
                               result=None,
                               details=None,
                               active_section="prediction")

    label, confidence = get_prediction(filepath)

    explanation_text = generate_explanation(label, confidence)

    # LOW CONFIDENCE
    if confidence < 60:
        details = None
        explanation_text += " ⚠️ Prediction confidence is low. Detailed disease information is hidden."
    else:
        details = get_disease_details(label)

    lime_img = explain_image(filepath)

    img_base64 = None
    if lime_img is not None:
        _, buffer = cv2.imencode(".jpg", (lime_img * 255).astype("uint8"))
        img_base64 = base64.b64encode(buffer).decode("utf-8")

    return render_template("index.html",
                           label=label,
                           confidence=min(confidence, 100),
                           result=img_base64,
                           explanation_text=explanation_text,
                           details=details,
                           active_section="prediction")

if __name__ == "__main__":
    app.run(debug=True)