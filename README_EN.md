# Receipt Helper

Local receipt OCR and intelligent classification tool powered by OpenVINO. All inference runs locally without cloud APIs, ensuring financial privacy.

## Features

- **Local OCR**: Extract structured text from receipt images using OpenVINO + PaddleOCR-VL
- **Intelligent Classification**: Classify, summarize, and organize receipts using a local small LLM (<35B)
- **Custom Rules**: Define custom classification categories and keywords
- **History**: Auto-save processing history with search support
- **Heterogeneous Compute**: Support CPU / GPU / NPU inference

## Requirements

- Python 3.10+
- Intel Core Ultra processor (recommended for NPU/GPU acceleration)

## Installation

```bash
pip install -r scripts/requirements.txt
```

## Quick Start

### 1. OCR Extraction

```bash
python scripts/openvino_ocr.py --input receipt.jpg
python scripts/openvino_ocr.py --input receipts_dir/ --device GPU
```

Outputs a JSON array with `file_name`, `ocr_text`, `amount`, `date`, `vendor`, etc.

### 2. LLM Classification

```bash
python scripts/receipt_llm.py --ocr-json ocr_result.json --instruction "按日期分类"
python scripts/receipt_llm.py --ocr-json ocr_result.json --instruction "统计餐饮总额"
```

### 3. Manage Classification Rules

```bash
python scripts/manage_rules.py list
python scripts/manage_rules.py add --category 培训 --keywords 培训,课程,讲座
python scripts/manage_rules.py history --limit 10
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PADDLEOCR_VL_DIR` | PaddleOCR-VL OpenVINO model directory | `../../openvino_notebooks/notebooks/paddleocr_vl` |
| `RECEIPT_LLM_MODEL_ID` | LLM model ID | `Qwen/Qwen2.5-7B-Instruct` |
| `RECEIPT_DATA_DIR` | Data storage directory (rules + history) | `./data` |

## Test Images

The `test_images/` directory contains sample receipt images for testing:

| Filename | Description |
|----------|-------------|
| `01_1.png` | Sample image 1 |
| `texi.jpg` | Taxi receipt |
| `texi2.jpg` | Taxi receipt 2 |
| `trip_02.jpg` | Travel receipt |

Quick OCR test:

```bash
python scripts/openvino_ocr.py --input test_images/
```

## Project Structure

```
receipt_helper/
├── SKILL.md                              # OpenVINO Skill definition
├── test_images/                          # Test images
│   ├── 01_1.png
│   ├── texi.jpg
│   ├── texi2.jpg
│   └── trip_02.jpg
├── data/
│   └── classification_rules.json         # Classification rules
└── scripts/
    ├── openvino_ocr.py                   # OCR script
    ├── receipt_llm.py                    # LLM analysis script
    ├── manage_rules.py                   # Rules/history management tool
    └── requirements.txt                  # Python dependencies
```

## License

MIT License
