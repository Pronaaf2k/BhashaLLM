
BanglaHandwritingOCR-Android

[![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://developer.android.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![Kotlin](https://img.shields.io/badge/Kotlin-7F52FF?style=for-the-badge&logo=kotlin&logoColor=white)](https://kotlinlang.org/)

An **offline, on-device** Bengali handwritten text recognition system for Android that combines computer vision with contextual language understanding using TinyLLMs.

> 🔬 **Research-Based**: Built upon the hybrid OCR approach from "*A hybrid approach to Bangla handwritten OCR: combining YOLO and an advanced CNN*" (Discover Artificial Intelligence, 2025)

🚀 Features

- **📱 Fully Offline**: No internet connection required after installation
- **🔍 Hybrid OCR Pipeline**: YOLOv8 + EfficientNet-B4 for accurate character detection and recognition
- **🧠 Context-Aware Correction**: TinyLLM-powered semantic understanding and grammar correction
- **🌐 Translation Ready**: Built-in English translation capabilities
- **⚡ Real-time Processing**: Optimized for mobile devices with 4-8GB RAM
- **📊 High Accuracy**: 93.87% grapheme root recognition, 98.22% diacritic recognition


 🏗️ Architecture

```
Image Capture → Preprocessing → YOLOv8 Detection → Character Isolation → 
EfficientNet Recognition → Word Formation → TinyLLM Correction → Final Output
```

## 📈 Performance

| Metric | Our System | Google Vision API |
|--------|------------|-------------------|
| **Character Error Rate** | **2.47%** | 13.89% |
| **Precision** | **97.01%** | 82.20% |
| **Recall** | **98.57%** | 96.53% |
| **Processing Time** | **<2s/word** | Cloud-dependent |

## 🛠️ Installation

### Prerequisites

- Android Studio Arctic Fox or later
- Android device with minimum 4GB RAM
- Android API level 24+ (Android 7.0)

### Build Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/BanglaHandwritingOCR-Android.git
   cd BanglaHandwritingOCR-Android
   ```

2. **Open in Android Studio**
   - Open Android Studio and select "Open an Existing Project"
   - Navigate to the cloned directory

3. **Build and Run**
   - Connect your Android device or start an emulator
   - Click "Run" or use `./gradlew installDebug`

### APK Download

[![Download APK](https://img.shields.io/badge/Download-APK-brightgreen?style=for-the-badge&logo=android)](https://github.com/your-username/BanglaHandwritingOCR-Android/releases/latest)

## 🎯 Usage

1. **Launch the app** on your Android device
2. **Capture image** using the camera or select from gallery
3. **Automatic processing** - the app will detect, recognize, and correct text
4. **View results** - see original OCR output and LLM-corrected version
5. **Export** - copy text or share results

## 📁 Project Structure

```
app/
├── src/main/
│   ├── java/com/banglaocr/
│   │   ├── ocr/              # OCR processing classes
│   │   ├── llm/              # TinyLLM integration
│   │   ├── camera/           # Camera handling
│   │   ├── ui/               # User interface
│   │   └── utils/            # Utility classes
│   ├── assets/
│   │   ├── models/           # ML models
│   │   └── datasets/         # Sample data
```

## 🧩 Models Used

| Model | Purpose | Size | Accuracy |
|-------|---------|------|----------|
| **YOLOv8-medium** | Character detection | ~25MB | 93.88% precision |
| **EfficientNet-B4** | Character recognition | ~19MB | 93.87% grapheme accuracy |
| **Phi-2 (INT8)** | Contextual correction | ~2.1GB | State-of-the-art |

## 📊 Dataset

This project uses multiple Bengali handwriting datasets:

- **CMATERdb** - Handwritten city names
- **BanglaLekha-Isolated** - 166,105 character images
- **BanglaWriting** - Multi-purpose paragraph dataset
- **Custom Collected Data** - 300 diverse handwritten words

## 🔧 Configuration

### Model Parameters

```kotlin
object OCRConfig {
    const val CONFIDENCE_THRESHOLD = 0.5f
    const val INPUT_SIZE = 640
    const val MAX_WORD_LENGTH = 20
    const val BATCH_SIZE = 4
}

object LLMConfig {
    const val CONTEXT_WINDOW = 2048
    const val TEMPERATURE = 0.7f
    const val MAX_TOKENS = 512
}
```

## 🚀 Performance Optimization

- **Quantization**: INT8 for CNN models, INT4 for LLM
- **NNAPI Acceleration**: Leverages device NPU/GPU
- **Model Caching**: Frequently used models kept in memory
- **Lazy Loading**: Models loaded on-demand
- **Background Processing**: Non-blocking UI operations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📝 Citation

If you use this project in your research, please cite the original paper:

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

## 🙏 Acknowledgments

- **Bangladesh University of Engineering and Technology (BUET)** for the foundational research
- **TensorFlow Lite** team for mobile ML optimization
- **Microsoft** for the Phi-2 model
- **All dataset contributors** and the Bengali NLP community

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🐛 Known Issues

- Processing time may vary based on device capabilities
- Very complex compound characters may require manual correction
- Limited support for cursive handwriting styles

---

<div align="center">

**Made with ❤️ for the Bengali-speaking community**

*Bringing AI accessibility to regional languages, one character at a time*

</div>
```

