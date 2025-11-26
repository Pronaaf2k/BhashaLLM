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

# --- Word Recognition Function ---
def recognize_word(image_paths):
    """
    Recognizes a word from a sequence of character images using the trained model.
    :param image_paths: List of paths to character images.
    :return: Recognized word.
    """
    recognized_word = ""
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Unable to load image at {image_path}. Skipping...")
            continue  # Skip invalid images

        # Preprocess the image
        try:
            image = cv2.resize(image, TARGET_IMAGE_SIZE)
            image = image.astype('float32') / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
            image = np.expand_dims(image, axis=0)   # Add batch dimension

            # Predict using the model
            predictions = model.predict(image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = index_to_class.get(predicted_class_index, "Unknown")

            recognized_word += predicted_class  # Append the character to the word
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue  # Skip on error

    return recognized_word

# --- Main Script ---
if __name__ == "__main__":
    # Example usage
    # List of character image paths (replace with actual paths to character images)
    character_image_paths = [
        r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\char1.png",
        r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\char2.png",
        r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\char3.png"
    ]

    recognized_word = recognize_word(character_image_paths)
    if recognized_word:
        print(f"Recognized Word: {recognized_word}")
    else:
        print("No valid characters were recognized.")