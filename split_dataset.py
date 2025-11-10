import os
import shutil
import random
from sklearn.model_selection import train_test_split

# --- Configuration ---
# !! IMPORTANT: Set this to the root of your preprocessed dataset !!
# This is where your character folders (e.g., 'অ', 'আ') are located.
PREPROCESSED_DATA_ROOT = r"C:\Users\chakl\OneDrive\Desktop\BhashaLLm\preprocessed_dataset"

# Define the target directories for the split dataset
OUTPUT_SPLIT_ROOT = os.path.join(os.path.dirname(PREPROCESSED_DATA_ROOT), "split_dataset")

# Define split ratios
TRAIN_RATIO = 0.70  # 70% for training
VAL_RATIO = 0.15    # 15% for validation
TEST_RATIO = 0.15   # 15% for testing (will be calculated as 1 - TRAIN_RATIO - VAL_RATIO)

RANDOM_SEED = 42    # For reproducible splits

# Accepted image file extensions
ACCEPTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# --- Script Start ---
if __name__ == '__main__':
    print("--- Starting Dataset Split Script ---")
    print(f"Reading from: {PREPROCESSED_DATA_ROOT}")
    print(f"Writing to: {OUTPUT_SPLIT_ROOT}")
    print(f"Train Ratio: {TRAIN_RATIO}, Validation Ratio: {VAL_RATIO}, Test Ratio: {TEST_RATIO}")

    # Validate ratios
    if not (TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0):
        print("\nWARNING: TRAIN_RATIO + VAL_RATIO + TEST_RATIO does not sum to 1.0!")
        print(f"Recalculating TEST_RATIO as 1 - {TRAIN_RATIO} - {VAL_RATIO}")
        TEST_RATIO = 1.0 - TRAIN_RATIO - VAL_RATIO
        print(f"New TEST_RATIO: {TEST_RATIO}")

    if not os.path.exists(PREPROCESSED_DATA_ROOT):
        print(f"\nERROR: The specified preprocessed data root DOES NOT EXIST:")
        print(f"       '{PREPROCESSED_DATA_ROOT}'")
        print("\n       Please ensure the path is correct and your preprocessing script has run successfully.")
        exit()

    # Clear and create output directories
    if os.path.exists(OUTPUT_SPLIT_ROOT):
        print(f"Output split directory '{OUTPUT_SPLIT_ROOT}' already exists. Clearing it...")
        shutil.rmtree(OUTPUT_SPLIT_ROOT)
    os.makedirs(os.path.join(OUTPUT_SPLIT_ROOT, 'train'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_SPLIT_ROOT, 'val'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_SPLIT_ROOT, 'test'), exist_ok=True)
    print(f"Created split dataset output root: {OUTPUT_SPLIT_ROOT}")


    all_image_paths = []
    all_labels = [] # Labels will be character names (string)
    class_names = []

    # 1. Collect all image paths and their labels
    print("\nCollecting image paths and labels...")
    for char_name in sorted(os.listdir(PREPROCESSED_DATA_ROOT)):
        char_dir = os.path.join(PREPROCESSED_DATA_ROOT, char_name)
        if os.path.isdir(char_dir):
            class_names.append(char_name)
            for img_name in os.listdir(char_dir):
                if img_name.lower().endswith(ACCEPTED_IMAGE_EXTENSIONS):
                    all_image_paths.append(os.path.join(char_dir, img_name))
                    all_labels.append(char_name) # Store character name as label

    if not all_image_paths:
        print(f"No images found in '{PREPROCESSED_DATA_ROOT}'. Please check the path and contents.")
        exit()

    print(f"Found {len(all_image_paths)} images across {len(class_names)} classes.")
    print(f"Classes found: {class_names}")

    # 2. Perform the initial train-test split (stratified)
    # The 'test_size' here is for the combined validation+test set.
    train_paths, val_test_paths, train_labels, val_test_labels = train_test_split(
        all_image_paths, all_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=all_labels
    )

    # 3. Split the remaining val_test_paths into validation and test sets (stratified)
    # Calculate the new test_size relative to the val_test_paths subset
    # e.g., if VAL_RATIO=0.15 and TEST_RATIO=0.15, then (0.15 / (0.15 + 0.15)) = 0.5
    test_size_relative = TEST_RATIO / (VAL_RATIO + TEST_RATIO) if (VAL_RATIO + TEST_RATIO) > 0 else 0

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_test_paths, val_test_labels,
        test_size=test_size_relative,
        random_state=RANDOM_SEED,
        stratify=val_test_labels
    )

    print(f"\nSplit distribution:")
    print(f"  Training set: {len(train_paths)} images")
    print(f"  Validation set: {len(val_paths)} images")
    print(f"  Test set: {len(test_paths)} images")

    # 4. Copy files to their respective new directories
    print("\nCopying files to their split directories (this may take a while for large datasets)...")

    def copy_files(paths, dataset_type):
        count = 0
        for img_path in paths:
            # Extract character name and image filename
            relative_path = os.path.relpath(img_path, PREPROCESSED_DATA_ROOT)
            char_name = os.path.dirname(relative_path)
            img_name = os.path.basename(img_path)

            # Create target subdirectory if it doesn't exist
            target_char_dir = os.path.join(OUTPUT_SPLIT_ROOT, dataset_type, char_name)
            os.makedirs(target_char_dir, exist_ok=True)

            # Copy the image
            shutil.copy(img_path, os.path.join(target_char_dir, img_name))
            count += 1
        print(f"  Copied {count} images to '{dataset_type}' directory.")

    copy_files(train_paths, 'train')
    copy_files(val_paths, 'val')
    copy_files(test_paths, 'test')

    print("\n--- Dataset Split Complete! ---")
    print(f"The split dataset is located at: {OUTPUT_SPLIT_ROOT}")
    print("You can now use the 'train', 'val', and 'test' subdirectories for model training.")