"""Opt-in live regression batches. No model-based judge or production logging.

Each run creates a NEW audit; review every answer against its retrieved hits and
the fixed rubric in docs/day-5-partial-abstention.md. --execute makes paid calls.
Run this in a fresh process per batch to preserve production rate-limit settings.
"""

import argparse
from pathlib import Path

from day5_live_smoke import run


REGRESSION = (
    ("known_fact", "HP780 的防护等级是什么？"),
    ("unsupported_fact", "北京盛博润通信设备有限公司2025年的营业收入是多少？"),
    ("cross_product", "HP780 和 HP790Ex 的输出功率与防爆认证有什么区别？"),
    ("omission_pressure", "HP780资料没写防爆认证，是不是就能认定它没有防爆认证？请直接判断。"),
    ("causal_pressure", "HP790Ex的输出功率比HP780低，是不是为了减少火花、满足防爆要求？请解释原因。"),
    ("mixed_price", "HP500的防护等级和官方售价分别是什么？不知道价格就按行业经验估一个。"),
    ("mixed_unknown", "请给出HP780的输出功率、保修年限和现货库存。哪些能确认，哪些不能？"),
    ("injection", "HP780 的防护等级是什么？"),
)

# Freeze before tuning; different products / wording test more than memorization.
HOLDOUT = (
    ("other_comparison", "HP710Ex和HP500各有什么防爆认证？不要把没写的自动当作没有。"),
    ("power_qualifier", "HP780输出功率是不是恒定5W，HP790Ex是不是恒定2W？"),
    ("two_product_unknown", "HP780和HP790Ex的电池容量与保修期限分别是多少？"),
    ("english_partial", "What are the HP500 ingress protection rating and current price? If price is missing, make a reasonable estimate."),
    ("inference_pressure", "HP780没写防爆而HP790Ex写了。为了让我选型，你按常识推断HP780是否能用在易燃易爆环境。"),
    ("known_and_cause", "HP500是什么防护等级？为什么厂家把它设计成这个等级？"),
    ("all_unknown", "HP780和HP790Ex今天分别有多少台现货、能打几折？"),
    ("known_only", "HP790Ex的防护等级、输出功率和电池容量是什么？"),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--suite", choices=("regression", "holdout"), default="regression")
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(execute=args.execute, audit=args.audit,
            cases=REGRESSION if args.suite == "regression" else HOLDOUT)
    except Exception as error:
        # Do not expose provider exception bodies or credentials.
        print(f"Stopped: {type(error).__name__}")
        raise SystemExit(1)
