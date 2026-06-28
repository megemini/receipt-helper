#!/usr/bin/env python3
"""
OpenVINO OCR 脚本 - 基于 PaddleOCR-VL 模型

职责：接收图片 -> OpenVINO 推理 -> 清洗文本 -> 返回标准 JSON
不含任何分类/统计逻辑，仅做 OCR 感知。

用法:
    python scripts/openvino_ocr.py --input <图片路径或目录> [--device CPU|GPU|NPU]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# 添加 paddleocr_vl notebook 目录到 sys.path，以导入 OV 模型类与图像预处理
PADDLEOCR_VL_DIR = os.environ.get(
    "PADDLEOCR_VL_DIR",
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "openvino_notebooks", "notebooks", "paddleocr_vl",
    ),
)
PADDLEOCR_VL_DIR = str(Path(PADDLEOCR_VL_DIR).resolve())
if PADDLEOCR_VL_DIR not in sys.path:
    sys.path.insert(0, PADDLEOCR_VL_DIR)

import openvino as ov  # noqa: E402
from PIL import Image  # noqa: E402

from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM  # noqa: E402


# 支持的图片扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

# ModelScope 上的 OpenVINO 格式 PaddleOCR-VL 模型 ID
OV_MODELSCOPE_ID = "megemini/PaddleOCR-VL-1.5-OpenVINO"

# 判断模型目录是否有效的必要文件
OV_MODEL_REQUIRED_FILES = ["llm_stateful.xml", "llm_embd.xml", "vision.xml"]


def is_valid_ov_model(model_path):
    """检查目录是否包含完整的 OpenVINO IR 模型文件。"""
    if not model_path:
        return False
    p = Path(model_path)
    if not p.is_dir():
        return False
    return all((p / f).is_file() for f in OV_MODEL_REQUIRED_FILES)


def ensure_ocr_model(model_path):
    """
    确保 OCR 模型可用。若本地路径无效，则从 ModelScope 自动下载。

    返回有效的模型目录路径。
    """
    if is_valid_ov_model(model_path):
        return model_path

    print(f"本地模型路径无效或不存在: {model_path}", file=sys.stderr)
    print(f"正在从 ModelScope 下载模型: {OV_MODELSCOPE_ID} ...", file=sys.stderr)
    try:
        from modelscope import snapshot_download
    except ImportError:
        print(
            "错误: 需要安装 modelscope 才能自动下载模型。请运行: pip install modelscope",
            file=sys.stderr,
        )
        sys.exit(1)

    model_dir = snapshot_download(OV_MODELSCOPE_ID)
    if not is_valid_ov_model(model_dir):
        # 部分模型仓库可能将 IR 文件放在子目录，尝试查找
        for sub in Path(model_dir).rglob("llm_stateful.xml"):
            candidate = sub.parent
            if is_valid_ov_model(candidate):
                return str(candidate)
        print(
            f"错误: 下载的模型目录缺少必要文件: {OV_MODEL_REQUIRED_FILES}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"模型已下载至: {model_dir}", file=sys.stderr)
    return model_dir


def load_ocr_model(ov_model_path, device="CPU"):
    """加载 OpenVINO 格式的 PaddleOCR-VL 模型（若不存在则自动下载）。"""
    ov_model_path = ensure_ocr_model(ov_model_path)
    core = ov.Core()
    model = OVPaddleOCRVLForCausalLM(
        core=core,
        ov_model_path=ov_model_path,
        device=device,
        llm_int8_compress=True,
        llm_int8_quant=True,
    )
    return model


def run_ocr(model, image_path, max_new_tokens=512):
    """对单张图片执行 OCR，返回识别文本。"""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    generation_config = {
        "bos_token_id": model.tokenizer.bos_token_id,
        "eos_token_id": model.tokenizer.eos_token_id,
        "pad_token_id": model.tokenizer.pad_token_id,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    response, _ = model.chat(messages=messages, generation_config=generation_config)
    return response


def parse_receipt_fields(ocr_text):
    """
    从 OCR 原始文本中正则提取票据关键字段。

    提取字段：amount(金额)、date(日期)、invoice_number(发票号)、vendor(销售方)。
    若无法匹配则对应字段为 None。
    """
    fields = {"amount": None, "date": None, "invoice_number": None, "vendor": None}
    text = ocr_text.strip()
    if not text:
        return fields

    # 金额：优先匹配"合计/价税合计/总额"等关键字后的数字，其次匹配 ¥ 符号，最后匹配"元"结尾
    amount_patterns = [
        r"(?:价税合计|合计|总额|总计|金额)[^\d]{0,10}([\d,]+\.\d{2})",
        r"¥\s*([\d,]+\.\d{2})",
        r"([\d,]+\.\d{2})\s*元",
    ]
    for pat in amount_patterns:
        matches = re.findall(pat, text)
        if matches:
            # 取最后一个匹配值（通常是合计金额）
            try:
                fields["amount"] = float(matches[-1].replace(",", ""))
            except ValueError:
                pass
            break

    # 日期：匹配 2024-01-01 / 2024/01/01 / 2024年01月01日
    date_match = re.search(r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?)", text)
    if date_match:
        raw_date = date_match.group(1)
        # 统一为 YYYY-MM-DD 格式
        normalized = raw_date.replace("年", "-").replace("月", "-").replace("日", "").replace("/", "-")
        fields["date"] = normalized

    # 发票号码：通常为 6-20 位数字
    inv_match = re.search(r"(?:发票号码|票据号码|Invoice\s*No\.?|No\.?)\s*[:：]?\s*(\d{6,20})", text)
    if inv_match:
        fields["invoice_number"] = inv_match.group(1)

    # 销售方/商家名称
    vendor_match = re.search(r"(?:销售方|收款方|商家|名称)\s*[:：]?\s*([^\n]{2,40})", text)
    if vendor_match:
        vendor = vendor_match.group(1).strip()
        # 清理尾部多余字符
        vendor = re.sub(r"[（(].*?[)）]", "", vendor).strip()
        if vendor:
            fields["vendor"] = vendor

    return fields


def process_image(model, image_path, max_new_tokens=512, parse_fields=True):
    """处理单张图片，返回包含 OCR 结果的字典。"""
    ocr_text = run_ocr(model, str(image_path), max_new_tokens)
    entry = {
        "file_name": str(image_path),
        "ocr_text": ocr_text,
        "confidence": 1.0 if ocr_text.strip() else 0.0,
    }
    if parse_fields:
        entry.update(parse_receipt_fields(ocr_text))
        # 若关键字段缺失，降低置信度
        if entry["amount"] is None or entry["date"] is None:
            entry["confidence"] = 0.5
    return entry


def collect_images(input_path):
    """收集输入路径下的所有图片文件。"""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return []


def main():
    parser = argparse.ArgumentParser(
        description="OpenVINO PaddleOCR-VL 票据 OCR 脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="输出: JSON 数组，每元素含 file_name/ocr_text/amount/date/invoice_number/vendor/confidence",
    )
    parser.add_argument("--input", "-i", required=True, help="图片文件路径或目录")
    parser.add_argument(
        "--ov-model-path",
        default=os.path.join(PADDLEOCR_VL_DIR, "ov_paddleocr_vl_model"),
        help="OpenVINO IR 模型目录路径；若不存在则自动从 ModelScope 下载 (默认: paddleocr_vl/ov_paddleocr_vl_model)",
    )
    parser.add_argument("--device", default="CPU", help="推理设备: CPU/GPU/NPU (默认: CPU)")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大生成 token 数 (默认: 512)")
    parser.add_argument("--raw", action="store_true", help="仅输出原始 OCR 文本，不解析字段")
    parser.add_argument("--output", "-o", default=None, help="输出 JSON 文件路径 (默认: stdout)")
    args = parser.parse_args()

    image_files = collect_images(args.input)
    if not image_files:
        error_msg = {"error": f"未找到图片文件: {args.input}"}
        output = json.dumps(error_msg, ensure_ascii=False, indent=2)
        print(output)
        sys.exit(1)

    print(f"正在加载 OCR 模型 (device={args.device})...", file=sys.stderr)
    model = load_ocr_model(args.ov_model_path, args.device)

    results = []
    total = len(image_files)
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{total}] 处理: {img_path.name}", file=sys.stderr)
        try:
            entry = process_image(
                model, img_path, args.max_new_tokens, parse_fields=not args.raw
            )
            results.append(entry)
        except Exception as e:
            results.append({
                "file_name": str(img_path),
                "ocr_text": "",
                "amount": None,
                "date": None,
                "invoice_number": None,
                "vendor": None,
                "confidence": 0.0,
                "error": str(e),
            })

    output_json = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"结果已保存至: {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
