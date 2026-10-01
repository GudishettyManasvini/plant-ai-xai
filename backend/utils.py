import numpy as np
import tensorflow as tf
import cv2
import json
import os

from lime import lime_image
from skimage.segmentation import mark_boundaries


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)


JSON_PATH = os.path.join(BASE_DIR, "class_names.json")

with open(JSON_PATH, "r") as f:
    class_names = json.load(f)


def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    return img


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


def explain_image(img_path):

    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    def predict_fn(images):
        images = np.array(images) / 255.0
        return model.predict(images)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img,
        predict_fn,
        top_labels=1,
        hide_color=None,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    return mark_boundaries(temp / 255.0, mask)


def generate_explanation(label, confidence):

    if "invalid" in label.lower():
        return "The uploaded image is not a valid plant leaf."

    if "healthy" in label.lower():
        return f"The plant is healthy with {confidence:.2f}% confidence."

    if confidence > 85:
        return f"The model strongly predicts {label.replace('_',' ')} with high confidence ({confidence:.2f}%). Disease regions are clearly visible."

    elif confidence > 60:
        return f"The model predicts {label.replace('_',' ')} with moderate confidence ({confidence:.2f}%). Some disease patterns are visible."

    else:
        return f"The prediction confidence is low ({confidence:.2f}%). Image quality or disease features may be unclear."


disease_info = {
    "Potato___Early_blight": {
        "description": "A fungal disease affecting potato leaves.",
        "cause": "Alternaria solani fungus.",
        "symptoms": "Brown spots with concentric rings.",
        "treatment": "Use fungicides and remove infected leaves."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Fungal disease causing orange-yellow spots.",
        "cause": "Gymnosporangium fungus.",
        "symptoms": "Orange spots on leaves.",
        "treatment": "Apply fungicide and remove infected parts."
    },
    "Healthy": {
        "description": "No disease detected.",
        "cause": "Healthy plant.",
        "symptoms": "Green leaf.",
        "treatment": "Maintain care."
    }
}

def get_disease_details(label):
    return disease_info.get(label, {
        "description": "No info available.",
        "cause": "Unknown",
        "symptoms": "Not identified",
        "treatment": "Consult expert."
    })