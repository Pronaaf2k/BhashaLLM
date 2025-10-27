import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
from gensim.models import Word2Vec

# --- Configuration ---
IMG_PATH = 'sample_bangla_word.jpg'
MODEL_DIR = 'models'
MAPPING_FILE = 'grapheme_maps.json'

# Placeholder file paths (assuming they exist in the defined structure)
YOLO_WEIGHTS = os.path.join(MODEL_DIR, 'mock_weights_detector.pt')
RECOGNIZER_WEIGHTS = os.path.join(MODEL_DIR, 'mock_weights_recognizer.pt')
WORD2VEC_MODEL = os.path.join(MODEL_DIR, 'mock_word2vec_corrector.pkl')


def load_assets():
    """Loads all necessary models and mappings."""
    print("--- 1. Loading Assets ---")
    
    # 1. Load Grapheme Mappings
    try:
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            grapheme_maps = json.load(f)
        print(f"Loaded grapheme mappings from {MAPPING_FILE}")
    except FileNotFoundError:
        print(f"ERROR: Mapping file {MAPPING_FILE} not found. Using empty map.")
        grapheme_maps = {"grapheme_root": {}, "vowel_diacritic": {}, "consonant_diacritic": {}}


    # 2. Load YOLOv8 Detector (Mock Initialization)
    try:
        # In a real scenario, we would load the model using YOLO(YOLO_WEIGHTS)
        detector = f"MOCK_YOLO_MODEL_LOADED: {YOLO_WEIGHTS}"
        print(f"Loaded Detector: {detector}")
    except Exception as e:
        print(f"MOCK ERROR loading YOLO: {e}")
        detector = None

    # 3. Load EfficientNet Recognizer (Mock Initialization)
    # We simulate loading the weights for our custom EfficientNet
    recognizer = f"MOCK_EFFICIENTNET_MODEL_LOADED: {RECOGNIZER_WEIGHTS}"
    print(f"Loaded Recognizer: {recognizer}")

    # 4. Load Word2Vec Corrector (Mock Initialization)
    try:
        # We simulate a gensim model, though the file content is just a placeholder
        # In reality: word2vec_model = Word2Vec.load(WORD2VEC_MODEL)
        word2vec_model = {"করদ": ["করদ", 0.99], "খরদ": ["করদ", 0.85], "গরা": ["গড়া", 0.90]}
        print(f"Loaded Word2Vec Corrector (Mock Dictionary): {WORD2VEC_MODEL}")
    except Exception:
        word2vec_model = {}

    return detector, recognizer, word2vec_model, grapheme_maps


def preprocess_image(img_path):
    """Loads and prepares the image for detection."""
    if not os.path.exists(img_path):
        print(f"ERROR: Image file not found at {img_path}. Using blank canvas.")
        img = np.zeros((256, 1024, 3), dtype=np.uint8) + 255
    else:
        # Load using OpenCV (standard for CV pipelines)
        img = cv2.imread(img_path)
    
    # Simple resizing/normalization steps would occur here
    print(f"Preprocessed image of size: {img.shape[:2]}")
    return img


def run_yolov8_detection(image_data):
    """Simulates running YOLOv8 to find character bounding boxes."""
    
    # Real logic: results = detector.predict(image_data)
    
    # --- SIMULATION: Detecting the word 'খরদ' (Khara-D) ---
    # We simulate 3 detections (x_min, y_min, x_max, y_max, confidence, class_id)
    # We assume reading left-to-right.
    
    simulated_detections = [
        {'box': [50, 50, 150, 150], 'patch_id': 0}, # Corresponds to 'খ' (Kh)
        {'box': [160, 50, 260, 150], 'patch_id': 1}, # Corresponds to 'র' (R)
        {'box': [270, 50, 370, 150], 'patch_id': 2}, # Corresponds to 'দ' (D)
    ]
    print(f"Detection successful. Found {len(simulated_detections)} character/grapheme patches.")
    return simulated_detections


def sort_and_extract_patches(image_data, detections):
    """Sorts detections and extracts character patches."""
    
    # 1. Sorting (usually complex, involving clustering by baseline, then sorting by x-coordinate)
    # Since simulated boxes are already LTR, we just verify the sort key (x_min)
    sorted_detections = sorted(detections, key=lambda d: d['box'][0])
    
    patches = []
    
    for i, det in enumerate(sorted_detections):
        x_min, y_min, x_max, y_max = det['box']
        # Extract patch (mock extraction, not actually slicing the blank canvas)
        patch = np.zeros((100, 100, 3), dtype=np.uint8) + 255
        
        # In a real scenario, patch = image_data[y_min:y_max, x_min:x_max]
        patches.append(patch)
        print(f"Extracted Patch {i}: {det['box']}")
        
    return patches


def run_efficientnet_recognition(patch, grapheme_maps):
    """
    Simulates the EfficientNet recognition, which outputs three classification targets
    per patch: Grapheme Root, Vowel Diacritic, and Consonant Diacritic.
    """
    
    # Real logic: 
    # output = recognizer(patch) 
    # root_id, v_diac_id, c_diac_id = output.argmax(dim=1)
    
    # --- SIMULATION: Using predefined class IDs for 'খরদ' ---
    # 1. 'খ' (Root ID 1, Vowel ID 0, Consonant ID 0)
    # 2. 'র' (Root ID 3, Vowel ID 0, Consonant ID 0)
    # 3. 'দ' (Root ID 4, Vowel ID 0, Consonant ID 0)
    
    simulated_recognition_outputs = [
        (1, 0, 0), # 'খ'
        (3, 0, 0), # 'র'
        (4, 0, 0)  # 'দ'
    ]
    
    recognized_characters = []
    
    for i, (r_id, v_id, c_id) in enumerate(simulated_recognition_outputs):
        # Ensure IDs are strings for JSON lookup
        r_id_str, v_id_str, c_id_str = str(r_id), str(v_id), str(c_id)
        
        root = grapheme_maps['grapheme_root'].get(r_id_str, '?')
        v_diac = grapheme_maps['vowel_diacritic'].get(v_id_str, '')
        c_diac = grapheme_maps['consonant_diacritic'].get(c_id_str, '')
        
        # Assemble the recognized character/unit (Root + Vowel Diacritic + Consonant Diacritic)
        unit = root + v_diac + c_diac
        recognized_characters.append(unit)
        
        print(f"Patch {i} recognized: Root={root} ({r_id}), Vowel={v_diac} ({v_id}), Consonant={c_diac} ({c_id}) -> Unit='{unit}'")

    return recognized_characters


def form_word(character_units):
    """Assembles the sequence of recognized character units into a full word string."""
    recognized_word = "".join(character_units)
    return recognized_word


def run_word2vec_correction(recognized_word, model):
    """
    Simulates using Word2Vec or a similar embedding model for spelling correction.
    In a true implementation, this might involve finding the nearest neighbor word in the embedding space
    that exists in a known dictionary, weighted by recognition confidence.
    """
    
    print(f"\n--- 4. Running Word2Vec Correction for: '{recognized_word}' ---")
    
    # If the recognized word is in our mock dictionary (simulating a known misspelling)
    if recognized_word in model:
        corrected_word, confidence = model[recognized_word]
        if confidence < 1.0:
            print(f"Correction applied: {recognized_word} -> {corrected_word} (Confidence: {confidence:.2f})")
            return corrected_word
        else:
            print("Word recognized correctly (high confidence). No correction needed.")
            return recognized_word
    
    # If the word is 'করদ' (assumed correct spelling)
    elif recognized_word == 'করদ':
        return recognized_word
        
    else:
        # Fallback for unrecognized words
        print("Word not found in mock correction dictionary. Returning original.")
        return recognized_word


def run_inference_pipeline(img_path):
    """Main function to run the entire OCR pipeline."""
    
    detector, recognizer, word2vec_model, grapheme_maps = load_assets()
    
    if not detector or not word2vec_model:
        print("\nFATAL ERROR: Failed to load models. Exiting.")
        return

    print("\n--- 2. Detection Stage (YOLOv8) ---")
    image_data = preprocess_image(img_path)
    detections = run_yolov8_detection(image_data)
    
    print("\n--- 3. Recognition Stage (EfficientNet) ---")
    patches = sort_and_extract_patches(image_data, detections)
    
    if not patches:
        print("No patches found. Cannot proceed to recognition.")
        return
        
    # We only need to run the recognition logic on the first patch (for simulation consistency)
    # In a real run, we run it on all patches:
    recognized_units = []
    for patch in patches:
        # Since the simulation only provides a full sequence of results, we skip looping 
        # and use the pre-defined results from the recognition function for simplicity.
        pass
    
    recognized_units = run_efficientnet_recognition(patches[0], grapheme_maps) # This function simulates the results for all 3 patches

    recognized_word = form_word(recognized_units)
    
    print(f"\nRecognized Character Units: {recognized_units}")
    print(f"Raw Recognized Word (Pre-Correction): '{recognized_word}'")
    
    # --- 4. Correction Stage (Word2Vec) ---
    final_corrected_word = run_word2vec_correction(recognized_word, word2vec_model)
    
    print("\n=============================================")
    print(f"FINAL INFERENCE RESULT")
    print(f"Recognized Word (Raw): {recognized_word}")
    print(f"Corrected Word (Final): {final_corrected_word}")
    print("=============================================")


if __name__ == "__main__":
    # Ensure the model directory exists for mock loading simulation
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create the mock files if they don't exist, so the script runs without failure
    for path, content in [
        (YOLO_WEIGHTS, b'Mock YOLO'),
        (RECOGNIZER_WEIGHTS, b'Mock Recognizer'),
        (WORD2VEC_MODEL, b'Mock Word2Vec')
    ]:
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(content)

    # To run this script successfully, ensure 'grapheme_maps.json' is in the root directory.
    # We simulate an input error: recognizing 'খরদ' (Khara-D), which Word2Vec corrects to 'করদ' (Korod).
    run_inference_pipeline(IMG_PATH)