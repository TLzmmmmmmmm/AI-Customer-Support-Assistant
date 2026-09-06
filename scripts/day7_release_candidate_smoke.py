"""One-shot local release-candidate smoke run for Day 7 Step 6.

The runner exercises the real FastAPI route with the configured embedding and
generation providers.  Its exclusive audit file prevents accidental reruns.
"""

from pathlib import Path

from day5_live_smoke import run


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "eval"
    / "results"
    / "day7-step6-release-candidate-smoke-attempt-2.jsonl"
)

CASES = (
    ("chinese_product_fact", "HP780 的防护等级是什么？"),
    ("chinese_product_capability", "网闸支持哪些安全邮件功能？"),
    ("product_recommendation", "多台接收机想共用一路射频线路，应该选什么设备？"),
    ("solution", "灾区电力和电信设施受损时，怎样快速建立宽带应急通信？"),
    ("unknown_company_fact", "北京盛博润通信设备有限公司 2026 年的营业收入是多少？"),
    ("hr1060_corrected_semantics", "HR1060 的电源电压是多少？"),
)


if __name__ == "__main__":
    run(
        execute=True,
        audit=AUDIT,
        # Attempt 1 completed the first case before the harness budget stopped.
        # Resume only the five unexecuted cases; never bill the completed case twice.
        cases=CASES[1:],
        max_prompt_utf8_bytes=50_000,
    )
