# PoRAG (Bangla RAG) Setup & Usage

> **Scope note.** This describes exploratory work **outside the paper's
> methodology.** Section III-D explains why retrieval-augmented generation
> was deliberately not used, and no reported result depends on this
> pipeline. Retained because Section VII lists retrieval as the first item
> of future work. See [`ERRATA.md`](ERRATA.md) §C5.


**Goal:** Turn your OCR text into a searchable knowledge base using Retrieval-Augmented Generation (RAG).

## 📂 Installation
We are using the `BhashaLLM` virtual environment (`venv`) for dependency management.

1.  **Dependencies Installed:**
    - `langchain`, `chromadb` (Vector Store), `sentence-transformers` (Embeddings)
    - `transformers`, `bitsandbytes`, `accelerate` (LLM Engine)

2.  **Repo Structure:**
    - `~/BhashaLLM/PoRAG/` (Source code)
    - `~/BhashaLLM/venv/` (Python Environment)

## 🚀 How to Run

### 1. Prepare Your Data
Save your OCR output (or any Bangla text) into a `.txt` file.
Example: `data/my_document.txt`

### 2. Run the Pipeline
Use the `venv` python to run `main.py` from the PoRAG directory.

```bash
# Navigate to project
cd ~/BhashaLLM

# Run RAG on your text file
./venv/bin/python PoRAG/main.py --text_path data/my_document.txt
```

### 3. Interact
The script will prompt you for a question in Bangla.
Example: * "রবীন্দ্রনাথ ঠাকুরের জন্মস্থান কোথায়?"*

## ⚙️ Configuration (Advanced)

You can swap the LLM or Embeddings model by passing arguments:

```bash
./venv/bin/python PoRAG/main.py \
  --text_path data/my_doc.txt \
  --chat_model "Qwen/Qwen2.5-1.5B-Instruct" \
  --quantization  # Use 4-bit loading for speed
```

### Recommended Models
- **Embedding:** `l3cube-pune/bengali-sentence-similarity-sbert` (Default)
- **LLM (Tiny):** `Qwen/Qwen2.5-1.5B-Instruct` (Fast, decent)
- **LLM (Strong):** `mistralai/Mistral-7B-Instruct-v0.3` (Slower, smarter)

## 🔧 Troubleshooting
- **Memory Error:** Add `--quantization` to use 4-bit mode.
- **"Repo not found":** Ensure you have internet access or the model is cached.
- **"ImportError":** Activate venv: `source venv/bin/activate`
