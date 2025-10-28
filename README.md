# Bangla Handwritten OCR Pipeline

This project implements a hybrid Optical Character Recognition (OCR) pipeline for handwritten Bangla words, as detailed in the research paper, "A hybrid approach to Bangla handwritten OCR: combining YOLO and an advanced CNN." The system leverages a multi-stage approach to accurately convert images of handwritten Bangla words into digital text.

The pipeline first uses a **YOLOv8** model to detect individual characters and graphemes within a word image. These detected character patches are then passed to a modified **EfficientNet-B4** model for recognition. The recognition model identifies three components for each character: the grapheme root, vowel diacritic, and consonant diacritic. Finally, an optional **Word2Vec**-based spelling correction layer refines the output, improving overall accuracy.

## Key Features

-   **Hybrid Model:** Combines deep learning models for detection (YOLOv8) and recognition (EfficientNet) for a robust pipeline.
-   **Handles Complex Scripts:** Specifically designed to manage the intricacies of the Bangla script, including compound characters (conjuncts), modifiers (matras), and diacritics.
-   **Two-Step Process:** Decouples character detection from recognition, allowing for specialized models to handle each task effectively.
-   **Spelling Correction:** An optional Word2Vec-based corrector enhances resilience by fixing common recognition errors.
-   **Extensible:** The modular design allows for future expansion, such as handling paragraph-level OCR.

## Directory Structure
