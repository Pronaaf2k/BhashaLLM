#!/usr/bin/env python3
import json
import textwrap
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(text).strip().splitlines(True)}


cells = [
    md(
        """
        # PaliGemma Bangla Line OCR Training + Graph Tracker

        Goal: train the existing PaliGemma OCR path on the line-by-line datasets under `/home/benaaf/Desktop/datasets`.

        This notebook tracks:

        - Dataset inventory and which datasets are real OCR training data.
        - Train/val/test split sizes.
        - Text length, image size, and source distribution graphs.
        - Training loss / eval loss graphs from `trainer_state.json`.
        - CER/WER evaluation graphs after inference.
        """
    ),
    code(
        """
        from pathlib import Path
        import csv, json, os, random, subprocess, sys
        from collections import Counter

        import matplotlib.pyplot as plt
        import pandas as pd
        from PIL import Image
        from IPython.display import display

        PROJECT_ROOT = Path('/home/benaaf/Desktop/BhashaLLM')
        DATASETS_ROOT = Path('/home/benaaf/Desktop/datasets')
        VLM_DIR = PROJECT_ROOT / 'training' / 'vlm_ocr'
        CONFIG_PATH = VLM_DIR / 'config.json'
        GRAPH_DIR = VLM_DIR / 'graphs'
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)

        os.chdir(PROJECT_ROOT)
        print('Project:', PROJECT_ROOT)
        print('Datasets:', DATASETS_ROOT)
        print('Graphs:', GRAPH_DIR)
        """
    ),
    md("""## 1. Dataset Inventory"""),
    code(
        """
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        meta_exts = {'.csv', '.tsv', '.json', '.jsonl', '.txt'}

        inventory = []
        for folder in sorted(p for p in DATASETS_ROOT.iterdir() if p.is_dir()):
            images = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in image_exts]
            metadata = [p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in meta_exts]
            inventory.append({'dataset': folder.name, 'images': len(images), 'metadata_files': len(metadata)})

        inv_df = pd.DataFrame(inventory).sort_values('images', ascending=False)
        display(inv_df)

        ax = inv_df.plot(kind='barh', x='dataset', y='images', figsize=(10, 5), legend=False, color='#2d7554')
        ax.set_title('Image Count By Dataset Folder')
        ax.set_xlabel('Images')
        ax.set_ylabel('Dataset')
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / 'dataset_image_counts.png', dpi=180)
        plt.show()
        """
    ),
    md(
        """
        ## 2. CSV Schema Discovery

        PaliGemma needs image-text pairs. The current direct line OCR candidates are:

        - `dataset10thMay.../dataset10thMay/dataset10thmay.csv` with `filename,text`.
        - `image_787 to image_927/combined_bhashallm_dataset.csv` with `File name,extracted text`.

        Other folders are still useful, but some are word-level, character-level, pix2pix cleanup, or unlabeled image folders.
        """
    ),
    code(
        """
        csv_rows = []
        for path in sorted(DATASETS_ROOT.rglob('*.csv')):
            try:
                with path.open('r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                csv_rows.append({
                    'path': str(path.relative_to(DATASETS_ROOT)),
                    'rows': len(rows),
                    'columns': ', '.join(reader.fieldnames or []),
                    'first_row': rows[0] if rows else {},
                })
            except Exception as exc:
                csv_rows.append({'path': str(path.relative_to(DATASETS_ROOT)), 'rows': None, 'columns': f'ERROR: {exc}', 'first_row': {}})

        csv_df = pd.DataFrame(csv_rows)
        display(csv_df[['path', 'rows', 'columns']])
        """
    ),
    md("""## 3. Select Training CSVs"""),
    code(
        """
        SELECTED_DATASETS = [
            {
                'name': 'bnaf_lines',
                'csv': DATASETS_ROOT / 'bnaf' / 'bnaf.csv',
                'image_root': DATASETS_ROOT / 'bnaf',
            },
            {
                'name': 'dataset10thMay_lines',
                'csv': DATASETS_ROOT / 'dataset10thMay-20260514T174723Z-3-001' / 'dataset10thMay' / 'dataset10thmay.csv',
                'image_root': DATASETS_ROOT / 'dataset10thMay-20260514T174723Z-3-001' / 'dataset10thMay',
            },
            {
                'name': 'image_787_927_lines',
                'csv': DATASETS_ROOT / 'image_787 to image_927' / 'combined_bhashallm_dataset.csv',
                'image_root': DATASETS_ROOT / 'image_787 to image_927',
            },
        ]

        for item in SELECTED_DATASETS:
            print(item['name'])
            print('  csv:', item['csv'], item['csv'].exists())
            print('  image_root:', item['image_root'], item['image_root'].exists())
        """
    ),
    md("""## 4. Preview Selected Data + Image/Text Stats"""),
    code(
        """
        def resolve_columns(columns):
            image_col = next((c for c in ['image', 'image_path', 'path', 'filename', 'file_name', 'File name'] if c in columns), None)
            text_col = next((c for c in ['text', 'label', 'transcription', 'extracted text', 'ground_truth', 'gt'] if c in columns), None)
            return image_col, text_col

        records = []
        for item in SELECTED_DATASETS:
            with item['csv'].open('r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                image_col, text_col = resolve_columns(reader.fieldnames or [])
                if not image_col or not text_col:
                    raise ValueError(f'Unsupported columns in {item["csv"]}: {reader.fieldnames}')
                for row in reader:
                    image_path = Path(row[image_col])
                    if not image_path.is_absolute():
                        image_path = item['image_root'] / image_path
                    text = (row[text_col] or '').strip()
                    if not text:
                        continue
                    exists = image_path.exists()
                    width = height = None
                    if exists:
                        try:
                            with Image.open(image_path) as im:
                                width, height = im.size
                        except Exception:
                            exists = False
                    records.append({
                        'source': item['name'],
                        'image': str(image_path),
                        'exists': exists,
                        'text': text,
                        'chars': len(text),
                        'words': len(text.split()),
                        'width': width,
                        'height': height,
                    })

        df = pd.DataFrame(records)
        display(df.groupby('source').agg(rows=('image', 'count'), valid_images=('exists', 'sum'), avg_chars=('chars', 'mean'), avg_words=('words', 'mean')))
        display(df.head(10))
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        df.groupby('source').size().plot(kind='bar', ax=axes[0], color='#2d7554')
        axes[0].set_title('Selected Line Samples By Source')
        axes[0].set_ylabel('Rows')

        df['chars'].hist(ax=axes[1], bins=30, color='#46505a')
        axes[1].set_title('Text Length Distribution')
        axes[1].set_xlabel('Characters per line')

        valid = df[df['exists'] & df['width'].notna() & df['height'].notna()]
        axes[2].scatter(valid['width'], valid['height'], s=12, alpha=0.6, color='#b4232c')
        axes[2].set_title('Image Size Scatter')
        axes[2].set_xlabel('Width')
        axes[2].set_ylabel('Height')

        plt.tight_layout()
        plt.savefig(GRAPH_DIR / 'selected_dataset_graphs.png', dpi=180)
        plt.show()
        """
    ),
    code(
        """
        sample_rows = df[df['exists']].sample(min(6, int(df['exists'].sum())), random_state=42)
        for _, row in sample_rows.iterrows():
            print(row['source'])
            print(row['text'])
            display(Image.open(row['image']))
        """
    ),
    md("""## 5. Build PaliGemma JSONL Splits"""),
    code(
        """
        # Build one JSONL split per selected source. Then merge them for the main training run.
        split_dirs = []
        for item in SELECTED_DATASETS:
            out_dir = VLM_DIR / 'data_desktop' / item['name']
            split_dirs.append(out_dir)
            cmd = [
                sys.executable, 'training/vlm_ocr/import_dataset.py',
                '--input', str(item['csv']),
                '--image-root', str(item['image_root']),
                '--out-dir', str(out_dir),
                '--train-ratio', '0.9',
                '--val-ratio', '0.05',
            ]
            print('Running:', ' '.join(cmd))
            result = subprocess.run(cmd, text=True, capture_output=True)
            print(result.stdout)
            print(result.stderr)
            if result.returncode != 0:
                raise RuntimeError('Import failed')
        """
    ),
    code(
        """
        merged_dir = VLM_DIR / 'data_desktop' / 'merged_lines'
        merged_dir.mkdir(parents=True, exist_ok=True)

        def read_jsonl(path):
            if not path.exists():
                return []
            return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

        for split in ['train', 'val', 'test']:
            rows = []
            for split_dir in split_dirs:
                rows.extend(read_jsonl(split_dir / f'{split}.jsonl'))
            random.Random(42).shuffle(rows)
            with (merged_dir / f'{split}.jsonl').open('w', encoding='utf-8') as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(split, len(rows), merged_dir / f'{split}.jsonl')
        """
    ),
    md("""## 6. Configure Existing PaliGemma Training"""),
    code(
        """
        cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        cfg.update({
            'base_model': 'google/paligemma2-3b-pt-448',
            'prompt': 'Read the handwritten Bangla text in this image. Return only the exact text. Preserve line breaks when visible.',
            'train_jsonl': 'training/vlm_ocr/data_desktop/merged_lines/train.jsonl',
            'val_jsonl': 'training/vlm_ocr/data_desktop/merged_lines/val.jsonl',
            'test_jsonl': 'training/vlm_ocr/data_desktop/merged_lines/test.jsonl',
            'output_dir': 'training/vlm_ocr/outputs/paligemma2_3b_448_desktop_lines_lora',
            'max_length': 2048,
            'epochs': 3,
            'batch_size': 1,
            'gradient_accumulation_steps': 8,
            'learning_rate': 1e-4,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'save_steps': 100,
            'eval_steps': 100,
            'logging_steps': 10,
            'max_new_tokens': 256,
        })
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        """
    ),
    md("""## 7. Validate Splits"""),
    code(
        """
        for split in ['train', 'val', 'test']:
            path = f'training/vlm_ocr/data_desktop/merged_lines/{split}.jsonl'
            cmd = [sys.executable, 'training/vlm_ocr/check_dataset.py', path]
            print('\nRunning:', ' '.join(cmd))
            result = subprocess.run(cmd, text=True, capture_output=True)
            print(result.stdout)
            print(result.stderr)
            if result.returncode != 0:
                raise RuntimeError(f'Validation failed for {split}')
        """
    ),
    md("""## 8. Train PaliGemma LoRA"""),
    code(
        """
        # Make sure Hugging Face access is ready before running:
        # huggingface-cli login
        # and accept the google/paligemma2-3b-pt-448 license.
        cmd = [sys.executable, 'training/vlm_ocr/train_paligemma_lora.py', '--config', str(CONFIG_PATH)]
        print('Running:', ' '.join(cmd))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        print('Return code:', process.returncode)
        """
    ),
    md("""## 9. Training Graphs From Trainer State"""),
    code(
        """
        output_dir = Path(json.loads(CONFIG_PATH.read_text(encoding='utf-8'))['output_dir'])
        state_paths = sorted(output_dir.glob('checkpoint-*/trainer_state.json'))
        state_path = state_paths[-1] if state_paths else output_dir / 'trainer_state.json'
        print('Using trainer state:', state_path)

        if state_path.exists():
            state = json.loads(state_path.read_text(encoding='utf-8'))
            history = pd.DataFrame(state.get('log_history', []))
            display(history.tail())

            fig, ax = plt.subplots(figsize=(9, 4))
            if 'loss' in history:
                history.dropna(subset=['loss']).plot(x='step', y='loss', ax=ax, label='train_loss', color='#2d7554')
            if 'eval_loss' in history:
                history.dropna(subset=['eval_loss']).plot(x='step', y='eval_loss', ax=ax, label='eval_loss', color='#b4232c')
            ax.set_title('PaliGemma Training / Eval Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            plt.tight_layout()
            plt.savefig(GRAPH_DIR / 'paligemma_training_loss.png', dpi=180)
            plt.show()
        else:
            print('No trainer_state.json found yet. Run training first.')
        """
    ),
    md("""## 10. Evaluate CER/WER"""),
    code(
        """
        cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        adapter = str(Path(cfg['output_dir']) / 'final_adapter')
        out_path = str(Path(cfg['output_dir']) / 'desktop_lines_test_predictions.jsonl')
        cmd = [
            sys.executable, 'training/vlm_ocr/evaluate_paligemma.py',
            '--config', str(CONFIG_PATH),
            '--jsonl', cfg['test_jsonl'],
            '--adapter', adapter,
            '--out', out_path,
        ]
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd, text=True, capture_output=True)
        print(result.stdout)
        print(result.stderr)
        """
    ),
    md("""## 11. Evaluation Graphs"""),
    code(
        """
        cfg = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        pred_path = Path(cfg['output_dir']) / 'desktop_lines_test_predictions.jsonl'
        if pred_path.exists():
            preds = pd.DataFrame([json.loads(line) for line in pred_path.read_text(encoding='utf-8').splitlines() if line.strip()])
            display(preds[['ref', 'pred', 'cer', 'wer']].head(10))
            print('Mean CER:', preds['cer'].mean())
            print('Mean WER:', preds['wer'].mean())

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            preds['cer'].hist(ax=axes[0], bins=20, color='#2d7554')
            axes[0].set_title('CER Distribution')
            preds['wer'].hist(ax=axes[1], bins=20, color='#46505a')
            axes[1].set_title('WER Distribution')
            plt.tight_layout()
            plt.savefig(GRAPH_DIR / 'paligemma_eval_cer_wer.png', dpi=180)
            plt.show()
        else:
            print('No predictions file found yet. Run evaluation first.')
        """
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).with_name("train_vlm_ocr.ipynb")
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
