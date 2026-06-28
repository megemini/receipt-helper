# Receipt Helper - 本地智能票据整理助手

基于 OpenVINO 的本地票据 OCR 识别与智能分类工具。所有推理在本地完成，无需云端 API，保障财务隐私安全。

## 功能特性

- **本地 OCR 识别**：基于 OpenVINO + PaddleOCR-VL，从票据图片中提取结构化文本
- **智能分类整理**：基于本地小型 LLM（<35B），根据用户指令自动分类、统计、归档
- **自定义分类规则**：支持用户自定义分类类别和关键词
- **历史记录**：自动保存整理历史，支持检索查询
- **异构算力**：支持 CPU / GPU / NPU 推理设备

## 环境要求

- Python 3.10+
- Intel 酷睿 Ultra 处理器（推荐，支持 NPU/GPU 加速）

## 安装

```bash
pip install -r scripts/requirements.txt
```

## 快速开始

### 1. OCR 识别

```bash
python scripts/openvino_ocr.py --input receipt.jpg
python scripts/openvino_ocr.py --input receipts_dir/ --device GPU
```

输出 JSON 数组，包含 `file_name`、`ocr_text`、`amount`、`date`、`vendor` 等字段。

### 2. LLM 分类整理

```bash
python scripts/receipt_llm.py --ocr-json ocr_result.json --instruction "按日期分类"
python scripts/receipt_llm.py --ocr-json ocr_result.json --instruction "统计餐饮总额"
```

### 3. 管理分类规则

```bash
python scripts/manage_rules.py list
python scripts/manage_rules.py add --category 培训 --keywords 培训,课程,讲座
python scripts/manage_rules.py history --limit 10
```

## 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `PADDLEOCR_VL_DIR` | PaddleOCR-VL OpenVINO 模型目录 | `../../openvino_notebooks/notebooks/paddleocr_vl` |
| `RECEIPT_LLM_MODEL_ID` | LLM 模型 ID | `Qwen/Qwen2.5-7B-Instruct` |
| `RECEIPT_DATA_DIR` | 数据存储目录（规则+历史） | `./data` |

## 测试图片

`test_images/` 目录包含测试用的票据图片：

| 文件名 | 说明 |
|--------|------|
| `01_1.png` | 测试图片 1 |
| `texi.jpg` | 出租车发票 |
| `texi2.jpg` | 出租车发票 2 |
| `trip_02.jpg` | 差旅票据 |

快速测试 OCR：

```bash
python scripts/openvino_ocr.py --input test_images/
```

## 项目结构

```
receipt_helper/
├── SKILL.md                              # OpenVINO Skill 定义
├── test_images/                          # 测试图片
│   ├── 01_1.png
│   ├── texi.jpg
│   ├── texi2.jpg
│   └── trip_02.jpg
├── data/
│   └── classification_rules.json         # 分类规则
└── scripts/
    ├── openvino_ocr.py                   # OCR 脚本
    ├── receipt_llm.py                    # LLM 分析脚本
    ├── manage_rules.py                   # 规则/历史管理工具
    └── requirements.txt                  # Python 依赖
```

## License

MIT License
