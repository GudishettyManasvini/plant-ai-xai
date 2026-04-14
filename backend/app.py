import os
import base64
import numpy as np
import cv2

from flask import Flask, render_template, request

from utils import get_prediction, explain_image, is_valid_plant_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------
# LANDING PAGE
# -----------------------
@app.route("/")
def landing():
    return render_template("landing.html")


# -----------------------
# DASHBOARD PAGE
# -----------------------
@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


# -----------------------
# PREDICTION + XAI
# -----------------------
@app.route("/explain", methods=["POST"])
def explain():

    file = request.files["image"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    # -----------------------
    # VALID IMAGE CHECK
    # -----------------------
    if not is_valid_plant_image(path):
        return render_template(
            "index.html",
            label="❌ Invalid Image (Not a Plant Leaf)",
            confidence=0,
            result=None,
            active_section="prediction"
        )

    # -----------------------
    # PREDICTION
    # -----------------------
    label, confidence = get_prediction(path)
    confidence = float(confidence)

    # -----------------------
    # LOW CONFIDENCE HANDLING
    # -----------------------
    if confidence < 50:
        label = "⚠ Uncertain / Irrelevant Leaf"
        confidence = confidence * 0.6

    # -----------------------
    # LIME EXPLANATION
    # -----------------------
    result = explain_image(path)

    img_base64 = None

    if result is not None:
        result = np.array(result)

        if result.max() <= 1.0:
            result = (result * 255).astype(np.uint8)
        else:
            result = result.astype(np.uint8)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', result)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

    return render_template(
        "index.html",
        label=label,
        confidence=round(confidence, 2),
        result=img_base64,
        active_section="prediction"
    )


if __name__ == "__main__":
    app.run(debug=True)