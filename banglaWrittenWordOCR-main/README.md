# Bangla Handwritten Word OCR

This repository contains an OCR pipeline tailored for recognizing handwritten Bangla words using a hybrid model. The pipeline combines YOLO for character detection and a ResNet-based CNN for recognition, with additional preprocessing and spelling correction for enhanced accuracy.

## Repository Structure

- **detection_model/**: YOLO model files for character detection.
- **recognition_model/**: ResNet-based model files for character recognition.
- **spellingCorrection_model/**: Implements spelling correction using a Word2Vec model.
- **data/BanglaGrapheme/**: Data files for training and testing.
- **weights2/**: Pre-trained weights for YOLO and ResNet models.
- **handw_training.ipynb**: Notebook for model training.
- **implementation_code.ipynb**: Main implementation notebook for running the OCR pipeline.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Salekin-13/banglaWrittenWordOCR.git
   cd banglaWrittenWordOCR
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision pillow gensim
   ```

3. **Download Pretrained Weights**:
   - Place YOLO weights in `detection_model/` and ResNet weights in `recognition_model/`.

## Usage

1. **Preprocess Images**:
   - Use the provided functions in `implementation_code.ipynb` to preprocess images by converting to grayscale, enhancing brightness, and adjusting contrast.

2. **Run the OCR Pipeline**:
   - Open and execute `implementation_code.ipynb`. This notebook loads the models, preprocesses input images, performs character detection and recognition, and applies spelling correction.

3. **Evaluation**:
   - Evaluate model performance using metrics such as Character Error Rate (CER) and Word Error Rate (WER), which are calculated within the notebook.

## Contributing

Contributions are welcome! Please open issues or submit pull requests to improve the pipeline.

## License

This project is licensed under the MIT License.
