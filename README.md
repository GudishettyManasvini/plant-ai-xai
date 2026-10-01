# Plant-AI-XAI

Plant-AI-XAI is an **AI-based plant disease detection application** that uses CNN-based image classification and Explainable AI to provide predictions along with visual explanations.

## Features

* **Plant Disease Detection** – Detects plant diseases from uploaded leaf images using a CNN model.
* **Prediction Confidence** – Provides confidence scores along with model predictions.
* **Explainable AI** – Uses **LIME** to generate visual explanations for model predictions.
* **Image Preprocessing** – Processes uploaded images using OpenCV before model inference.
* **Secure Image Upload** – Validates uploaded files and applies size and content checks.
* **Safe File Handling** – Uses randomized server-side filenames and temporary-file cleanup.

## Tech Stack

* **Language:** Python
* **Backend:** Flask
* **Machine Learning:** CNN, NumPy, Pandas
* **Computer Vision:** OpenCV
* **Explainable AI:** LIME

## Security

* Validates uploaded image files before processing.
* Applies a **5 MB file-size limit**.
* Performs content validation on uploaded files.
* Uses randomized filenames for server-side storage.
* Cleans up temporary files after processing.
* Reduces the risk of unsafe or oversized file uploads.

## Running Locally

1. Clone the repository.
2. Install the required Python dependencies.
3. Configure the project environment.
4. Start the Flask application.
5. Open the application in your browser and upload a leaf image for prediction.
