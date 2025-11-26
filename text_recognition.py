import os
import tensorflow as tf
import numpy as np
import cv2
import json

# --- Configuration ---
MODEL_PATH = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\bangla_ocr_model.h5"
CLASS_INDICES_PATH = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\class_indices.json"
TARGET_IMAGE_SIZE = (32, 32)  # Must match the training configuration

# --- Load Model and Class Indices ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading class indices...")
with open(CLASS_INDICES_PATH, 'r') as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}  # Reverse mapping

print(f"Loaded {len(index_to_class)} classes.")

# --- Text Recognition Function ---
def recognize_text(image_path):
    """
    Recognizes text from a single image using the trained model.
    :param image_path: Path to the input image.
    :return: Recognized text.
    """
    # Load and preprocess the image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    image = cv2.resize(image, TARGET_IMAGE_SIZE)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    # Predict using the model
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = index_to_class.get(predicted_class_index, "Unknown")

    print(f"Predicted class: {predicted_class}")
    return predicted_class

# --- Main Script ---
if __name__ == "__main__":
    # Example usage
    test_image_path = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\test_image.png"  # Replace with your test image path
    recognized_text = recognize_text(test_image_path)
    if recognized_text:
        print(f"Recognized Text: {recognized_text}")