#!/usr/bin/env python3
"""
分类规则与历史记录管理工具

用法:
    # 查看当前所有分类规则
    python scripts/manage_rules.py list

    # 添加新类别（关键词用逗号分隔）
    python scripts/manage_rules.py add --category 培训 --keywords 培训,课程,讲座 --description 培训费用

    # 删除某个类别
    python scripts/manage_rules.py remove --category 培训

    # 重置为默认规则
    python scripts/manage_rules.py reset

    # 查看历史记录（最近 N 条）
    python scripts/manage_rules.py history --limit 10

    # 搜索历史记录
    python scripts/manage_rules.py history --search 餐饮
"""
import argparse
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    "RECEIPT_DATA_DIR",
    Path(__file__).resolve().parent.parent / "data",
))
RULES_FILE = DATA_DIR / "classification_rules.json"
HISTORY_FILE = DATA_DIR / "history.jsonl"

# 默认分类规则（用于 reset）
DEFAULT_RULES = {
    "version": "1.0",
    "default_category": "待确认",
    "rules": [
        {"category": "餐饮", "keywords": ["餐", "食", "饭店", "外卖", "咖啡", "茶饮", "饮料", "小吃", "酒楼", "食堂"], "description": "餐饮消费"},
        {"category": "交通", "keywords": ["打车", "出租", "地铁", "公交", "加油", "停车", "高铁", "机票", "车票", "滴滴", "出行"], "description": "交通出行"},
        {"category": "办公用品", "keywords": ["文具", "打印", "办公", "纸张", "耗材", "复印"], "description": "办公用品采购"},
        {"category": "住宿", "keywords": ["住宿", "宾馆", "民宿", "旅馆", "客栈", "房费"], "description": "差旅住宿"},
        {"category": "通讯", "keywords": ["话费", "流量", "通讯", "电信", "移动", "联通", "宽带"], "description": "通讯费用"},
        {"category": "差旅", "keywords": ["差旅", "出差", "报销", "补贴"], "description": "差旅费用"},
        {"category": "医疗", "keywords": ["医院", "药店", "医药", "门诊", "挂号", "体检"], "description": "医疗健康"},
        {"category": "娱乐", "keywords": ["电影", "游戏", "娱乐", "健身", "运动", "KTV"], "description": "娱乐休闲"},
    ],
}


# ========== 规则管理 ==========

def load_rules():
    if not RULES_FILE.is_file():
        return None
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def save_rules(rules):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RULES_FILE.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_list(args):
    rules = load_rules()
    if not rules:
        print(f"规则文件不存在: {RULES_FILE}")
        print("请先运行: python scripts/manage_rules.py reset")
        return
    print(f"默认类别: {rules.get('default_category', '待确认')}")
    print(f"共 {len(rules.get('rules', []))} 个分类:\n")
    for r in rules.get("rules", []):
        kws = "、".join(r.get("keywords", []))
        print(f"  [{r['category']}] {r.get('description', '')}")
        print(f"    关键词: {kws}")
    print(f"\n规则文件路径: {RULES_FILE}")


def cmd_add(args):
    rules = load_rules()
    if not rules:
        rules = {"version": "1.0", "default_category": "待确认", "rules": []}
    rules.setdefault("rules", [])

    # 检查是否已存在
    for r in rules["rules"]:
        if r["category"] == args.category:
            print(f"类别 '{args.category}' 已存在，请使用 update 或先 remove。")
            sys.exit(1)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    new_rule = {
        "category": args.category,
        "keywords": keywords,
        "description": args.description or "",
    }
    rules["rules"].append(new_rule)
    save_rules(rules)
    print(f"已添加类别: {args.category}")
    print(f"  关键词: {'、'.join(keywords)}")
    print(f"  描述: {args.description or '(无)'}")


def cmd_remove(args):
    rules = load_rules()
    if not rules:
        print(f"规则文件不存在: {RULES_FILE}")
        sys.exit(1)
    original_len = len(rules.get("rules", []))
    rules["rules"] = [r for r in rules.get("rules", []) if r["category"] != args.category]
    if len(rules["rules"]) == original_len:
        print(f"未找到类别: {args.category}")
        sys.exit(1)
    save_rules(rules)
    print(f"已删除类别: {args.category}")


def cmd_reset(args):
    save_rules(json.loads(json.dumps(DEFAULT_RULES)))  # 深拷贝
    print(f"已重置为默认规则 ({len(DEFAULT_RULES['rules'])} 个类别)")
    print(f"规则文件: {RULES_FILE}")


# ========== 历史记录管理 ==========

def cmd_history(args):
    if not HISTORY_FILE.is_file():
        print(f"历史记录文件不存在: {HISTORY_FILE}")
        print("运行 receipt_llm.py 后会自动生成历史记录。")
        return

    lines = HISTORY_FILE.read_text(encoding="utf-8").strip().split("\n")
    records = []
    for line in lines:
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # 搜索过滤
    if args.search:
        keyword = args.search.lower()
        records = [
            r for r in records
            if keyword in r.get("instruction", "").lower()
            or keyword in r.get("result", "").lower()
            or any(keyword in str(v).lower() for v in [s.get("vendor", "") for s in r.get("ocr_summary", [])])
        ]

    # 限制条数（取最近 N 条）
    limit = args.limit or len(records)
    records = records[-limit:]

    if not records:
        print("没有匹配的历史记录。")
        return

    print(f"共 {len(records)} 条历史记录 (显示最近 {limit} 条):\n")
    print("-" * 70)
    for i, r in enumerate(records, 1):
        print(f"[{i}] 时间: {r.get('timestamp', '?')}")
        print(f"    指令: {r.get('instruction', '?')}")
        print(f"    票据数: {r.get('ocr_count', 0)}  总金额: ¥{r.get('total_amount', 0):.2f}")
        # 展示前 3 条票据摘要
        for s in r.get("ocr_summary", [])[:3]:
            print(f"      - {s.get('date', '?')} | {s.get('vendor', '?')} | ¥{s.get('amount', '?')}")
        if len(r.get("ocr_summary", [])) > 3:
            print(f"      ... 共 {len(r['ocr_summary'])} 张")
        if args.full:
            print(f"    结果:\n{r.get('result', '')}")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="分类规则与历史记录管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="查看当前所有分类规则")

    # add
    p_add = sub.add_parser("add", help="添加新的分类类别")
    p_add.add_argument("--category", required=True, help="类别名称")
    p_add.add_argument("--keywords", required=True, help="关键词，逗号分隔 (如: 餐,食,饭店)")
    p_add.add_argument("--description", default="", help="类别描述")

    # remove
    p_remove = sub.add_parser("remove", help="删除某个分类类别")
    p_remove.add_argument("--category", required=True, help="要删除的类别名称")

    # reset
    sub.add_parser("reset", help="重置为默认分类规则")

    # history
    p_hist = sub.add_parser("history", help="查看历史整理记录")
    p_hist.add_argument("--limit", type=int, default=10, help="显示最近 N 条 (默认: 10)")
    p_hist.add_argument("--search", default=None, help="按关键词搜索历史记录")
    p_hist.add_argument("--full", action="store_true", help="显示完整结果内容")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "reset":
        cmd_reset(args)
    elif args.command == "history":
        cmd_history(args)


if __name__ == "__main__":
    main()
