# AGENTS.md

## Project Overview

Local receipt OCR + classification skill using OpenVINO. Three Python scripts, no tests, no CI, no build system.

- `scripts/openvino_ocr.py` — OCR extraction from receipt images (OpenVINO + PaddleOCR-VL)
- `scripts/receipt_llm.py` — LLM-based classification/analysis of OCR results (OpenVINO + Qwen3.5-35B-A3B)
- `scripts/manage_rules.py` — CLI tool for classification rules and history

## Key Commands

```bash
# Install dependencies
pip install -r scripts/requirements.txt

# OCR: extract text from receipts
python scripts/openvino_ocr.py --input <path> [--device CPU|GPU|NPU]

# LLM: classify/analyze OCR results
python scripts/receipt_llm.py --ocr-json <ocr.json> --instruction "按日期分类"

# Manage classification rules
python scripts/manage_rules.py list
python scripts/manage_rules.py add --category 培训 --keywords 培训,课程
python scripts/manage_rules.py history --limit 10
```

## Architecture Notes

- All推理 runs locally via OpenVINO (CPU/GPU/NPU). No cloud APIs.
- `openvino_ocr.py` depends on `ov_paddleocr_vl` module from a sibling `openvino_notebooks/` repo (see `PADDLEOCR_VL_DIR` env var, defaults to `../../openvino_notebooks/notebooks/paddleocr_vl`).
- OCR model auto-downloads from ModelScope (`megemini/PaddleOCR-VL-1.5-OpenVINO`) if local path invalid.
- LLM model defaults to `Qwen/Qwen3.5-35B-A3B` (35B MoE, 3B active), configurable via `RECEIPT_LLM_MODEL_ID` env var.
- Classification rules live in `data/classification_rules.json`, history in `data/history.jsonl`.

## Gotchas

- The `scripts/ov_paddleocr_vl_model/` directory exists but is empty — model files must be downloaded first.
- OCR script adds the OpenVINO notebooks directory to `sys.path` at import time; the notebooks repo must exist at the expected relative path or `PADDLEOCR_VL_DIR` must be set.
- LLM script reads OCR data from stdin by default if `--ocr-json` is omitted.
- The `--no-history` flag on `receipt_llm.py` skips writing to `history.jsonl`.
- SKILL.md is the OpenCode Skill definition, not developer documentation. Refer to scripts' `--help` for usage.
