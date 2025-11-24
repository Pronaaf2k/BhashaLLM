

# **BanglaHandwritingOCR-Android**

[![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge\&logo=android\&logoColor=white)](https://developer.android.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/lite)
[![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge\&logo=kotlin\&logoColor=white)](https://kotlinlang.org/)
[![Live Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge\&logo=vercel\&logoColor=white)](https://bhashallmhandwritingrecognitionsyst.vercel.app/)

An **offline, on-device Bengali handwritten text recognition system for Android** combining computer vision and contextual language understanding using TinyLLMs.

> 🔬 **Research-Based:** Built upon the hybrid OCR approach from
> *“A hybrid approach to Bangla handwritten OCR: combining YOLO and an advanced CNN”* (Discover Artificial Intelligence, 2025)

---

## 🚀 Features

* 📱 **Fully Offline** — no internet required
* 🔍 **Hybrid OCR**: YOLOv8 + EfficientNet-B4
* 🧠 **Context-Aware Correction** using TinyLLM
* 🌐 **Built-in Translation**
* ⚡ **Real-time Processing**
* 🎯 **High Accuracy**

  * 93.87% grapheme root recognition
  * 98.22% diacritic recognition

---

## 🏗️ Architecture

```
Image Capture → Preprocessing → YOLOv8 Detection → Character Isolation →
EfficientNet Recognition → Word Formation → TinyLLM Correction → Final Output
```

---

## 📈 Performance

| Metric                   | Our System   | Google Vision API |
| ------------------------ | ------------ | ----------------- |
| **Character Error Rate** | **2.47%**    | 13.89%            |
| **Precision**            | **97.01%**   | 82.20%            |
| **Recall**               | **98.57%**   | 96.53%            |
| **Processing Time**      | **<2s/word** | Cloud-dependent   |

---

## 🛠️ Installation

### **Prerequisites**

* Android Studio Arctic Fox+
* Android device with ≥4GB RAM
* Android API 24+ (Android 7.0)

### **Build Steps**

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/BanglaHandwritingOCR-Android.git
   cd BanglaHandwritingOCR-Android
   ```

2. **Open in Android Studio**

   * Select **Open an Existing Project**
   * Choose the cloned directory

3. **Build and Run**

   * Connect device or use emulator
   * Press **Run** or run:

     ```bash
     ./gradlew installDebug
     ```

### **APK Download**

[![Download APK](https://img.shields.io/badge/Download-APK-brightgreen?style=for-the-badge\&logo=android)](https://github.com/your-username/BanglaHandwritingOCR-Android/releases/latest)

---

## 🎯 Usage

1. Launch the app
2. Capture an image or import from gallery
3. OCR runs automatically
4. View raw + LLM-corrected output
5. Export or share the text

---

## 🔗 Live Demo

Try the online version here:
**[https://bhashallmhandwritingrecognitionsyst.vercel.app/](https://bhashallmhandwritingrecognitionsyst.vercel.app/)**

---

## 📁 Project Structure

```
app/
├── src/main/
│   ├── java/com/banglaocr/
│   │   ├── ocr/          # OCR processing classes
│   │   ├── llm/          # TinyLLM integration
│   │   ├── camera/       # Camera handling
│   │   ├── ui/           # User interface
│   │   └── utils/        # Utility classes
│   ├── assets/
│       ├── models/       # ML models
│       └── datasets/     # Sample data
```

---

## 🧩 Models Used

| Model               | Purpose               | Size   | Accuracy                 |
| ------------------- | --------------------- | ------ | ------------------------ |
| **YOLOv8-medium**   | Character detection   | ~25MB  | 93.88% precision         |
| **EfficientNet-B4** | Character recognition | ~19MB  | 93.87% grapheme accuracy |
| **Phi-2 (INT8)**    | Contextual correction | ~2.1GB | SOTA LLM                 |

---

## 📊 Dataset Sources

* **CMATERdb** — Handwritten city names
* **BanglaLekha-Isolated** — 166,105 characters
* **BanglaWriting** — Paragraph-level handwriting
* **Custom Dataset** — 300 additional samples

---

## 🔧 Configuration

### **OCR Parameters**

```kotlin
object OCRConfig {
    const val CONFIDENCE_THRESHOLD = 0.5f
    const val INPUT_SIZE = 640
    const val MAX_WORD_LENGTH = 20
    const val BATCH_SIZE = 4
}
```

### **LLM Parameters**

```kotlin
object LLMConfig {
    const val CONTEXT_WINDOW = 2048
    const val TEMPERATURE = 0.7f
    const val MAX_TOKENS = 512
}
```

---

## 🚀 Performance Optimization

* INT8/INT4 **quantization**
* **NNAPI Acceleration**
* **Model Caching**
* **Lazy Loading**
* **Background Processing**

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes
4. Push the branch
5. Create a Pull Request

---

## 📝 Citation

```bibtex
@article{maung2025hybrid,
  title={A hybrid approach to Bangla handwritten OCR: combining YOLO and an advanced CNN},
  author={Maung, Aye T. and Salekin, Sumaiya and Haque, Mohammad A.},
  journal={Discover Artificial Intelligence},
  volume={5},
  number={119},
  year={2025},
  publisher={Springer}
}
```

---

## 🙏 Acknowledgments

* **BUET** — Foundational OCR research
* **TensorFlow Lite** — Mobile ML optimization
* **Microsoft** — Phi-2 model
* Bengali NLP community & dataset contributors

---

## 📄 License

Licensed under the **MIT License**.

---

## 🐛 Known Issues

* Processing speed varies by device
* Some compound characters may require manual fixes
* Limited support for cursive handwriting

---

<div align="center">

**Made with ❤️ for the Bengali-speaking community**
*Bringing AI accessibility to regional languages, one character at a time.*

</div>

---
