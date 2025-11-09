import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil # For creating and clearing directories

# --- Preprocessing Function ---
def preprocess_image_for_ocr(image_path, target_size=(32, 32)):
    """
    Preprocesses a single image containing a Bangla character for OCR.

    Steps include:
    1. Load image in grayscale.
    2. Apply Gaussian blur for noise reduction.
    3. Perform adaptive binarization (background black, text white).
    4. Resize to a target_size.
    5. Normalize pixel values to the range [0, 1].
    6. Expand dimensions to (height, width, 1) for neural network input.

    Args:
        image_path (str): The full path to the input image file.
        target_size (tuple, optional): The desired output size (width, height)
                                       for the preprocessed image. Defaults to (32, 32).

    Returns:
        np.ndarray: A preprocessed image array of shape (target_size[0], target_size[1], 1)
                    with float32 pixel values in the range [0, 1].
                    Returns None if the image cannot be loaded.
    """
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        # print(f"Error: Could not load image from {image_path}. Skipping.")
        return None

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Noise Reduction (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Binarization using Adaptive Thresholding
    # For OCR, typically text is dark on a light background or vice-versa.
    # We want text to be white (255) and background black (0) for consistency
    # and often better feature extraction in neural networks.
    # THRESH_BINARY_INV means pixels > threshold become 0, others become 255.
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    # 4. Resize to target_size
    resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)

    # 5. Normalize Pixel Values to [0, 1]
    normalized = resized.astype('float32') / 255.0

    # 6. Expand dimensions to (height, width, 1) for Keras/TensorFlow compatibility
    preprocessed_image = np.expand_dims(normalized, axis=-1)

    return preprocessed_image

# --- DEMO / DATASET PROCESSING BLOCK ---
if __name__ == '__main__':
    print("--- Running Dataset Preprocessing Script ---")

    # --- Configuration ---
    # !! IMPORTANT: Update this path to your 'Images' folder !!
    # Example: If 'dataset2' is in C:\Users\chakl\OneDrive\Desktop\BhashLLm\,
    # then the path will be:
    BASE_DATASET_DIR = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\dataset2\BanglaLekha-Isolated\Images"

    # !! IMPORTANT: Define the directory where preprocessed images will be saved !!
    # This will create a 'preprocessed_dataset' folder at the same level as 'dataset2'
    # or you can put it anywhere else, e.g., inside 'BhashLLm' folder.
    OUTPUT_BASE_DIR = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\preprocessed_dataset"

    DEMO_TARGET_SIZE = (32, 32) # The size your TinyML model will expect
    ACCEPTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff') # Add more if needed

    # --- Setup Output Directory ---
    if os.path.exists(OUTPUT_BASE_DIR):
        print(f"Output directory '{OUTPUT_BASE_DIR}' already exists. Clearing it...")
        shutil.rmtree(OUTPUT_BASE_DIR) # Remove existing directory and its contents
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True) # Create a fresh output directory
    print(f"Created output directory: {OUTPUT_BASE_DIR}")

    # --- Iterate and Preprocess Dataset ---
    total_images_processed = 0
    total_images_skipped = 0
    sample_display_count = 0
    MAX_SAMPLES_TO_DISPLAY = 3 # Display only a few preprocessed images as a sanity check

    if not os.path.exists(BASE_DATASET_DIR):
        print(f"\nERROR: The specified base dataset directory DOES NOT EXIST:")
        print(f"       '{BASE_DATASET_DIR}'")
        print("\n       Please ensure the path is correct.")
    else:
        print(f"\nStarting to process images from: '{BASE_DATASET_DIR}'")
        print(f"Preprocessed images will be saved to: '{OUTPUT_BASE_DIR}'")
        print(f"Target size for preprocessing: {DEMO_TARGET_SIZE}\n")

        # Walk through the directory structure
        for root, dirs, files in os.walk(BASE_DATASET_DIR):
            # Construct the corresponding output subdirectory path
            relative_path = os.path.relpath(root, BASE_DATASET_DIR)
            output_subdir = os.path.join(OUTPUT_BASE_DIR, relative_path)
            os.makedirs(output_subdir, exist_ok=True)

            for file_name in files:
                if file_name.lower().endswith(ACCEPTED_IMAGE_EXTENSIONS):
                    image_path = os.path.join(root, file_name)
                    output_image_path = os.path.join(output_subdir, file_name)

                    preprocessed_output = preprocess_image_for_ocr(image_path, target_size=DEMO_TARGET_SIZE)

                    if preprocessed_output is not None:
                        # Save the preprocessed image
                        # cv2.imwrite expects a 0-255 image for saving to standard formats (PNG, JPG)
                        # So, we convert back to uint8 and scale by 255.
                        # It also expects 2D (for grayscale) or 3D (for color) but not 3D with channel of 1
                        # So, we squeeze to remove the last dimension.
                        cv2.imwrite(output_image_path, (preprocessed_output.squeeze() * 255).astype(np.uint8))
                        total_images_processed += 1

                        # Display a few sample preprocessed images
                        if sample_display_count < MAX_SAMPLES_TO_DISPLAY:
                            plt.figure(figsize=(2, 2))
                            plt.imshow(preprocessed_output.squeeze(), cmap='gray')
                            plt.title(f"Processed Sample {sample_display_count+1}")
                            plt.axis('off')
                            plt.show()
                            sample_display_count += 1

                    else:
                        total_images_skipped += 1
                        print(f"Skipped: {image_path} (could not load or process)")

    print(f"\n--- Preprocessing Summary ---")
    print(f"Total images processed and saved: {total_images_processed}")
    print(f"Total images skipped: {total_images_skipped}")
    print(f"Preprocessed images stored in: '{OUTPUT_BASE_DIR}'")
    if sample_display_count > 0:
        print(f"Displayed {sample_display_count} sample preprocessed images.")

    print("\n--- End of Script ---")