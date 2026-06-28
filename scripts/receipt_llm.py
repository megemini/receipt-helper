#!/usr/bin/env python3
"""
票据整理与分析 LLM 脚本 - 基于 OpenVINO LLM

职责：接收 OCR JSON 数据 + 用户自然语言指令 -> 本地 LLM 推理 -> 输出整理/统计结果

默认模型: Qwen/Qwen3.5-35B-A3B (35B MoE，激活参数 3B，中文能力优秀)
可通过环境变量 RECEIPT_LLM_MODEL_ID 替换为其他模型，例如:
  - Qwen/Qwen2.5-14B-Instruct
  - meta-llama/Llama-3.1-8B-Instruct

用法:
    python scripts/receipt_llm.py --ocr-json ocr_result.json --instruction "按日期分类并统计总金额"
    cat ocr_result.json | python scripts/receipt_llm.py --instruction "统计餐饮总额"
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# 默认 LLM 模型 ID（35B MoE，激活参数 3B）
DEFAULT_LLM_MODEL_ID = os.environ.get(
    "RECEIPT_LLM_MODEL_ID",
    "Qwen/Qwen3.5-35B-A3B",
)

# 数据存储目录（分类规则 + 历史记录）
DATA_DIR = Path(os.environ.get(
    "RECEIPT_DATA_DIR",
    Path(__file__).resolve().parent.parent / "data",
))
RULES_FILE = DATA_DIR / "classification_rules.json"
HISTORY_FILE = DATA_DIR / "history.jsonl"

SYSTEM_PROMPT = """你是一个专业的财务票据整理助手。你将收到通过 OCR 识别的票据结构化数据（JSON 格式）以及用户的整理指令。

你必须严格遵守以下规则：
1. **数据源唯一性**：所有金额、日期、商家等信息必须且只能来源于提供的 OCR JSON 数据，严禁凭空编造或"心算"出数据中不存在的数字。
2. **数学准确性**：进行求和、统计时，请逐项累加，确保计算正确。对不确定的计算结果，请展示计算过程。
3. **分类规则**：严格按照"用户自定义分类规则"进行分类。当商家名称或 OCR 文本命中某类别的关键词时归入该类别；若无法匹配任何规则，归入默认类别。
4. **结构化输出**：使用 Markdown 表格展示整理结果，并在表格后给出汇总信息（如总计金额）。
5. **诚实反馈**：若 OCR 数据中存在缺失字段（amount/date 为 null）或置信度低（confidence < 0.6），需明确指出哪些票据需要人工核对。
"""


def load_classification_rules(rules_path):
    """加载用户自定义分类规则。"""
    p = Path(rules_path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"警告: 分类规则文件解析失败 ({e})，将使用默认分类。", file=sys.stderr)
        return None


def rules_to_prompt(rules):
    """将分类规则格式化为 LLM 可读的文本。"""
    if not rules or not rules.get("rules"):
        return "（未提供自定义分类规则，请按常见财务类别自行判断。）"

    default_cat = rules.get("default_category", "待确认")
    lines = [f"默认类别：{default_cat}", "分类规则："]
    for r in rules["rules"]:
        kws = "、".join(r.get("keywords", []))
        desc = r.get("description", "")
        lines.append(f"- {r['category']}：关键词 [{kws}]（{desc}）")
    return "\n".join(lines)


def save_history(instruction, ocr_data, result, data_dir):
    """将本次整理结果追加到历史记录文件 (JSONL 格式)。"""
    history_path = Path(data_dir) / "history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # 精简存储的 OCR 摘要（避免历史文件膨胀）
    ocr_summary = [
        {
            "file_name": item.get("file_name", ""),
            "amount": item.get("amount"),
            "date": item.get("date"),
            "vendor": item.get("vendor"),
            "invoice_number": item.get("invoice_number"),
        }
        for item in ocr_data
    ]

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "instruction": instruction,
        "ocr_count": len(ocr_data),
        "total_amount": sum(
            r["amount"] for r in ocr_summary if r["amount"] is not None
        ),
        "ocr_summary": ocr_summary,
        "result": result,
    }

    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_llm(model_id, device="CPU"):
    """加载 OpenVINO 优化的小型 LLM 模型。"""
    from optimum.intel import OVModelForCausalLM
    from transformers import AutoTokenizer

    print(f"正在加载 LLM 模型: {model_id} (device={device})...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = OVModelForCausalLM.from_pretrained(
        model_id,
        device=device,
        trust_remote_code=True,
        ov_config={"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"},
    )
    return model, tokenizer


def build_user_prompt(ocr_data, instruction, rules_text=""):
    """构建传递给 LLM 的用户提示词。"""
    # 精简 OCR 数据，仅保留 LLM 推理所需字段
    receipts = []
    for item in ocr_data:
        receipt = {
            "文件名": item.get("file_name", ""),
            "金额": item.get("amount"),
            "日期": item.get("date"),
            "发票号": item.get("invoice_number"),
            "商家": item.get("vendor"),
            "OCR文本": item.get("ocr_text", "")[:300],  # 截断过长的 OCR 文本
            "置信度": item.get("confidence", 1.0),
        }
        if "error" in item:
            receipt["错误"] = item["error"]
        receipts.append(receipt)

    rules_section = ""
    if rules_text:
        rules_section = f"""
## 用户自定义分类规则
{rules_text}
"""

    prompt = f"""以下是 {len(receipts)} 张票据的 OCR 识别结果（JSON 格式）：
{rules_section}
```json
{json.dumps(receipts, ensure_ascii=False, indent=2)}
```

用户指令：{instruction}

请根据上述 OCR 数据和用户指令，完成整理与分析任务。输出要求：
- 使用 Markdown 表格展示结果（列：日期、商家/项目、金额、归属类别、备注）
- 表格后附上汇总信息（如总计金额、分类统计等）
- 若有票据字段缺失或置信度低，在备注中标注"建议人工核对"
- 若提供了自定义分类规则，必须严格按照规则中的关键词进行归类
"""
    return prompt


def generate(model, tokenizer, messages, max_new_tokens=2048):
    """调用 LLM 生成回复。"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    # 截取新生成的 token（去除输入部分）
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def read_ocr_json(ocr_json_arg):
    """从文件或 stdin 读取 OCR JSON 数据。"""
    if ocr_json_arg and ocr_json_arg != "-":
        content = Path(ocr_json_arg).read_text(encoding="utf-8")
    else:
        content = sys.stdin.read()
    data = json.loads(content)
    # 兼容单个对象或数组
    if isinstance(data, dict):
        data = [data]
    return data


def main():
    parser = argparse.ArgumentParser(
        description="票据整理与分析 LLM 脚本 (OpenVINO LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
        "  python receipt_llm.py --ocr-json ocr.json --instruction '按日期分类'\n"
        "  cat ocr.json | python receipt_llm.py --instruction '统计总金额' --output result.md\n"
        "  python receipt_llm.py --ocr-json ocr.json -p '按类别统计' --rules data/classification_rules.json",
    )
    parser.add_argument("--ocr-json", default="-", help="OCR JSON 文件路径 (默认: 从 stdin 读取)")
    parser.add_argument("--instruction", "-p", required=True, help="用户的整理/分析指令")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_LLM_MODEL_ID,
        help=f"LLM 模型 ID (默认: {DEFAULT_LLM_MODEL_ID})",
    )
    parser.add_argument("--device", default="CPU", help="推理设备: CPU/GPU/NPU (默认: CPU)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="最大生成 token 数 (默认: 2048)")
    parser.add_argument("--output", "-o", default=None, help="输出文件路径 (默认: stdout)")
    parser.add_argument(
        "--rules",
        default=str(RULES_FILE),
        help=f"自定义分类规则 JSON 文件路径 (默认: {RULES_FILE})",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="不将本次结果写入历史记录",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help=f"数据存储目录 (规则+历史) (默认: {DATA_DIR})",
    )
    args = parser.parse_args()

    # 读取 OCR 数据
    ocr_data = read_ocr_json(args.ocr_json)
    print(f"已加载 {len(ocr_data)} 张票据的 OCR 数据", file=sys.stderr)

    # 加载自定义分类规则
    rules = load_classification_rules(args.rules)
    rules_text = rules_to_prompt(rules) if rules else ""
    if rules:
        print(f"已加载分类规则: {len(rules.get('rules', []))} 个类别", file=sys.stderr)

    # 加载 LLM
    model, tokenizer = load_llm(args.model_id, args.device)

    # 构建对话
    user_prompt = build_user_prompt(ocr_data, args.instruction, rules_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 生成结果
    print("正在生成整理结果...", file=sys.stderr)
    response = generate(model, tokenizer, messages, args.max_new_tokens)

    if args.output:
        Path(args.output).write_text(response, encoding="utf-8")
        print(f"结果已保存至: {args.output}", file=sys.stderr)
    else:
        print(response)

    # 保存历史记录
    if not args.no_history:
        record = save_history(args.instruction, ocr_data, response, args.data_dir)
        print(f"历史记录已保存至: {Path(args.data_dir) / 'history.jsonl'}", file=sys.stderr)


if __name__ == "__main__":
    main()
