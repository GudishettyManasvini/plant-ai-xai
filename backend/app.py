from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
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

from security import save_secure_image


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Maximum request size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# ERROR HANDLER
# -----------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    return render_template(
        "index.html",
        label="File Too Large",
        confidence=0,
        explanation_text="Please upload an image smaller than 5 MB.",
        result=None,
        details=None,
        active_section="prediction"
    ), 413


# -----------------------------
# LANDING PAGE
# -----------------------------
@app.route("/")
def landing():
    return render_template("landing.html")


# -----------------------------
# DASHBOARD
# -----------------------------
@app.route("/dashboard")
def dashboard():
    return render_template(
        "index.html",
        active_section="upload"
    )


# -----------------------------
# PREDICTION
# -----------------------------
@app.route("/explain", methods=["POST"])
def explain():

    filepath = None

    try:
        file = request.files.get("image")

        if not file or not file.filename:
            return render_template(
                "index.html",
                label="No file uploaded",
                confidence=0,
                explanation_text="Please upload an image.",
                result=None,
                details=None,
                active_section="prediction"
            )

        # Securely validate and save the uploaded image.
        filepath = save_secure_image(
            file,
            UPLOAD_FOLDER
        )

        # Validate that the image is relevant to the application.
        if not is_valid_plant_image(filepath):
            return render_template(
                "index.html",
                label="Invalid Image",
                confidence=10,
                explanation_text="The uploaded image does not appear to contain a plant leaf.",
                result=None,
                details=None,
                active_section="prediction"
            )

        # Run ML prediction.
        label, confidence = get_prediction(filepath)

        explanation_text = generate_explanation(
            label,
            confidence
        )

        # LOW CONFIDENCE
        if confidence < 60:
            details = None

            explanation_text += (
                " ⚠️ Prediction confidence is low. "
                "Detailed disease information is hidden."
            )
        else:
            details = get_disease_details(label)

        # Generate XAI explanation.
        lime_img = explain_image(filepath)

        img_base64 = None

        if lime_img is not None:
            _, buffer = cv2.imencode(
                ".jpg",
                (lime_img * 255).astype("uint8")
            )

            img_base64 = base64.b64encode(
                buffer
            ).decode("utf-8")

        return render_template(
            "index.html",
            label=label,
            confidence=min(confidence, 100),
            result=img_base64,
            explanation_text=explanation_text,
            details=details,
            active_section="prediction"
        )

    except ValueError as error:

        return render_template(
            "index.html",
            label="Invalid Upload",
            confidence=0,
            explanation_text=str(error),
            result=None,
            details=None,
            active_section="prediction"
        )

    finally:
        # Delete temporary uploaded image after processing.
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


if __name__ == "__main__":
    # Debug mode should not be enabled for production.
    app.run(debug=False)