import numpy as np
import tensorflow as tf
import cv2
import json
import os

from lime import lime_image
from skimage.segmentation import mark_boundaries

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("models/plant_model.h5")

# -----------------------------
# LOAD CLASS NAMES (FIXED)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
json_path = os.path.join(BASE_DIR, "class_names.json")

with open(json_path, "r") as f:
    class_names = json.load(f)

# -----------------------------
# PREPROCESS IMAGE
# -----------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    return img


# -----------------------------
# CHECK IF PLANT IMAGE
# -----------------------------
def is_valid_plant_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])

    return green_ratio > 0.05


# -----------------------------
# PREDICTION FUNCTION (FOR LIME)
# -----------------------------
def predict_fn(images):
    images = np.array(images)
    return model.predict(images)


# -----------------------------
# GET PREDICTION + CONFIDENCE
# -----------------------------
def get_prediction(img_path):
    img = preprocess_image(img_path)

    if img is None:
        return "Invalid Image", 0

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]

    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    if class_index >= len(class_names):
        return "Unknown", confidence

    return class_names[class_index], confidence


# -----------------------------
# LIME EXPLANATION
# -----------------------------
def explain_image(img_path):
    img = preprocess_image(img_path)

    if img is None:
        return None

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image=img,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=300
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    return mark_boundaries(temp, mask)