import os
import uuid
import cv2
import numpy as np

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def is_allowed_extension(filename):
    """Check whether the uploaded file has an allowed image extension."""
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def save_secure_image(file, upload_folder):
    """
    Validate and securely save an uploaded image.

    Security controls:
    - Reject missing files
    - Restrict file size
    - Restrict file extensions
    - Validate actual image content
    - Generate a random server-side filename
    """

    if file is None or not file.filename:
        raise ValueError("No file was uploaded.")

    if not is_allowed_extension(file.filename):
        raise ValueError("Only JPG, JPEG and PNG images are allowed.")

    # Read uploaded content into memory so we can validate it
    file_data = file.read()

    if not file_data:
        raise ValueError("The uploaded file is empty.")

    if len(file_data) > MAX_FILE_SIZE:
        raise ValueError("File size exceeds the 5 MB limit.")

    # Validate that the file is actually a readable image.
    image_array = np.frombuffer(file_data, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded file is not a valid image.")

    # Generate a random filename instead of trusting user input.
    random_name = f"{uuid.uuid4().hex}.jpg"

    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, random_name)

    # Save the validated image in a normalized format.
    success = cv2.imwrite(filepath, image)

    if not success:
        raise ValueError("Unable to save the uploaded image.")

    return filepath