import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
        print(f"Error: Could not load image from {image_path}. Please check the path and file type.")
        return None

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Noise Reduction (Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Binarization using Adaptive Thresholding
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

# --- DEMO / TESTING BLOCK ---
if __name__ == '__main__':
    print("--- Running Preprocessing Demo for a Specific Image ---")

    # >>> IMPORTANT: YOU MUST FIND AN ACTUAL IMAGE FILE IN THIS DIRECTORY <<<
    # Your base directory is:
    BASE_IMAGE_DIR = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\dataset2\BanglaLekha-Isolated\Images\1"

    # Now, you need to append the name of an actual image file found inside that directory.
    # For example, if you have 'img_001.jpg' inside it, the full path would be:
    # IMAGE_PATH_TO_TEST = os.path.join(BASE_IMAGE_DIR, "img_001.jpg")
    #
    # PLEASE REPLACE "your_actual_image_file.png" with the correct filename and extension.
    # You can open the folder in File Explorer to see the filenames.
    IMAGE_PATH_TO_TEST = os.path.join(BASE_IMAGE_DIR, "1_1.jpg") # <--- **CHANGE "1_1.jpg" to your specific image filename**
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    DEMO_TARGET_SIZE = (32, 32) # The size your TinyML model will expect

    # --- Verify the Path Before Processing ---
    if not os.path.exists(IMAGE_PATH_TO_TEST):
        print(f"\nERROR: The specified image file does NOT EXIST:")
        print(f"       '{IMAGE_PATH_TO_TEST}'")
        print("\n       Please ensure you have:")
        print("       1. **UNZIPPED** 'dataset2.zip' correctly.")
        print("       2. **UPDATED** `IMAGE_PATH_TO_TEST` in the script to point to an **ACTUAL IMAGE FILE** (e.g., `...\\Images\\1\\your_image.png`)")
        print("       3. Checked for typos in the file name and extension within the script.")
    else:
        print(f"\nAttempting to preprocess image from: '{IMAGE_PATH_TO_TEST}'")
        print(f"Target size for preprocessing: {DEMO_TARGET_SIZE}")

        # Call the preprocessing function
        preprocessed_output = preprocess_image_for_ocr(IMAGE_PATH_TO_TEST, target_size=DEMO_TARGET_SIZE)

        # Display the results
        if preprocessed_output is not None:
            print("\n--- Preprocessing Successful! ---")
            print(f"Output type: {type(preprocessed_output)}")
            print(f"Output shape: {preprocessed_output.shape}")
            print(f"Pixel value range: [{np.min(preprocessed_output):.4f}, {np.max(preprocessed_output):.4f}]")

            # Visualize the preprocessed image
            plt.figure(figsize=(4, 4))
            # .squeeze() removes the channel dimension (e.g., from 32x32x1 to 32x32) for display purposes
            plt.imshow(preprocessed_output.squeeze(), cmap='gray')
            plt.title(f"Preprocessed Character ({DEMO_TARGET_SIZE[0]}x{DEMO_TARGET_SIZE[1]})")
            plt.axis('off')
            plt.show()
            print("\nDemo complete. The preprocessed image is displayed.")
        else:
            print("\n--- Preprocessing Demo Failed ---")
            print("Could not get a preprocessed output. Check error messages above.")

    print("\n--- End of Script ---")