import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import json # To save the class_indices mapping

# --- Configuration ---
# !! IMPORTANT: Set this to the root of your SPLIT dataset !!
# This should point to the directory created by split_dataset.py,
# e.g., 'C:\Users\chakl\OneDrive\Desktop\BhashLLm\split_dataset'
SPLIT_DATASET_ROOT = r"C:\Users\chakl\OneDrive\Desktop\BhashLLm\split_dataset"

# Target image size (must match your preprocessing script)
TARGET_IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 1) # (Height, Width, Channels)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50 # You can adjust this based on validation performance
LEARNING_RATE = 0.001

# Model saving path
MODEL_SAVE_PATH = os.path.join(os.path.dirname(SPLIT_DATASET_ROOT), "bangla_ocr_model.h5")
CLASS_INDICES_SAVE_PATH = os.path.join(os.path.dirname(SPLIT_DATASET_ROOT), "class_indices.json")

# --- Script Start ---
if __name__ == '__main__':
    print("--- Starting Model Training Script ---")
    print(f"Loading data from: {SPLIT_DATASET_ROOT}")
    print(f"Input image shape: {INPUT_SHAPE}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    print(f"Class indices will be saved to: {CLASS_INDICES_SAVE_PATH}")

    # Check if the split dataset root exists
    if not os.path.exists(SPLIT_DATASET_ROOT):
        print(f"\nERROR: The specified SPLIT dataset root DOES NOT EXIST:")
        print(f"       '{SPLIT_DATASET_ROOT}'")
        print("\n       Please ensure your split_dataset.py script ran successfully and created this directory.")
        exit()
    if not os.path.isdir(os.path.join(SPLIT_DATASET_ROOT, 'train')) or \
       not os.path.isdir(os.path.join(SPLIT_DATASET_ROOT, 'val')) or \
       not os.path.isdir(os.path.join(SPLIT_DATASET_ROOT, 'test')):
        print(f"\nERROR: The '{SPLIT_DATASET_ROOT}' directory does not contain 'train', 'val', and 'test' subdirectories.")
        print("       Please ensure the dataset splitting was completed correctly.")
        exit()

    # --- Data Generators ---
    # Your preprocessed images were saved as uint8 (0-255).
    # ImageDataGenerator's rescale argument will normalize them to [0, 1] as floats.
    datagen = ImageDataGenerator(rescale=1./255)

    print("\nSetting up data generators...")
    train_generator = datagen.flow_from_directory(
        os.path.join(SPLIT_DATASET_ROOT, 'train'),
        target_size=TARGET_IMAGE_SIZE,
        color_mode='grayscale', # Important for single channel images
        batch_size=BATCH_SIZE,
        class_mode='categorical', # For one-hot encoded labels
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        os.path.join(SPLIT_DATASET_ROOT, 'val'),
        target_size=TARGET_IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # No need to shuffle validation data
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(SPLIT_DATASET_ROOT, 'test'),
        target_size=TARGET_IMAGE_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # No need to shuffle test data
    )

    num_classes = train_generator.num_classes
    print(f"Number of classes detected: {num_classes}")
    print(f"Class indices: {train_generator.class_indices}")

    if num_classes == 0:
        print("ERROR: No classes/images found by the data generator. Please check your dataset structure.")
        exit()

    # Save the class_indices mapping for later inference
    with open(CLASS_INDICES_SAVE_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Class indices mapping saved to: {CLASS_INDICES_SAVE_PATH}")


    # --- Model Architecture ---
    print("\nDefining CNN model architecture...")
    model = Sequential([
        # First Convolutional Block
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2), # Add dropout for regularization

        # Second Convolutional Block
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # Flatten the 3D output to 1D for the Dense layers
        Flatten(),

        # Fully Connected Layers
        Dense(64, activation='relu'), # A relatively small number of neurons
        Dropout(0.3),

        # Output layer: 'softmax' for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', # Use this for one-hot encoded labels
                  metrics=['accuracy'])

    model.summary()

    # --- Callbacks ---
    # ModelCheckpoint: Save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # EarlyStopping: Stop training if validation accuracy doesn't improve for a few epochs
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1)

    # ReduceLROnPlateau: Reduce learning rate when validation accuracy stops improving
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=0.00001, verbose=1)

    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    # --- Train the Model ---
    print("\nStarting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=callbacks_list
    )

    print("\n--- Training Complete! ---")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Class indices mapping saved to: {CLASS_INDICES_SAVE_PATH}")

    # --- Evaluate the Model on Test Set ---
    print("\nEvaluating model on the test set...")
    # Load the best saved model for final evaluation
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)

    print(f"\n--- Test Set Evaluation ---")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\n--- Script Finished ---")