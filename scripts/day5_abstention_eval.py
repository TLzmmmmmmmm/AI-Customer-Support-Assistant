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

# Follow-up regression: language follows the actual question/preference, while
# referrals must not imply the existence of unprovided price pages or lists.
LANGUAGE_REFERRAL = (
    ("english_partial", "What are the HP500 ingress protection rating and current price? If price is missing, make a reasonable estimate."),
    ("english_known", "What is the ingress protection rating of the HP780?"),
    ("english_unknown", "How many HP780 units are in stock today, and what is the warranty period?"),
    ("explicit_english", "请用英语回答：HP500的防护等级和官方售价是什么？"),
    ("explicit_chinese", "Please answer in Chinese: what are the HP500 ingress protection rating and current price?"),
    ("mixed_price", "HP500的防护等级和官方售价分别是什么？不知道价格就按行业经验估一个。"),
    ("quote_page", "HP500官方售价在哪里能查？你们官网有公开报价或价格表吗？"),
    ("chinese_known", "HP780 的防护等级是什么？"),
)

REFERRAL_HOLDOUT = (
    ("quote_link", "给我HP500官网价格表的链接，不需要具体价格。"),
    ("english_quote_link", "Please give me the link to the HP500 price list on your website. If it is not in your information, say so."),
    ("false_quote_premise", "既然你们官网有HP500公开报价页，请告诉我页面地址和价格。"),
    ("catalog_link", "HP780的防护等级是什么？再给我官方产品目录PDF的下载链接。"),
    ("contact_zh", "公司的公开值班电话和电子邮箱是什么？"),
    ("contact_en", "Please give the company's public phone number and email address."),
    ("english_rating_cause", "What is the HP500 IP rating, and why did its manufacturer choose that rating?"),
    ("english_comparison", "Compare the HP780 and HP790Ex output power and explosion-protection certifications."),
)

# Owner-supplied identity: freeze before editing the production identity prompt.
# Exact brand/legal-name spelling and English-only answers are required; ordinary
# English style is not graded unless it changes facts or obscures abstention.
COMPANY_IDENTITY = (
    ("brand_en", "What is your company's official English brand name?"),
    ("full_name_en", "What is your company's full official English name?"),
    ("identity_zh", "请分别给出公司的英文品牌名和正式英文全称。"),
    ("incorrect_name", "Is your official company name Beijing Shengboyuan Communication Equipment Co., Ltd.? Please give the correct full English name and brand."),
    ("english_unknown", "How many HP780 units are in stock today, and what is the warranty period?"),
    ("contact_identity", "Please give your full English company name, public phone number and email address."),
    ("english_rating_cause", "What is the HP500 IP rating, and why did its manufacturer choose that rating?"),
    ("brand_quote_link", "What is your English brand name? Also give me the link to the HP500 price list on your website, or say clearly if you cannot confirm it."),
)

# Owner-confirmed radio suffix classification. See day-5-radio-policy.md for
# updated expectations of older regression cases; historical audits stay intact.
RADIO_POLICY = (
    ("plain_radio", "HP780是不是防爆对讲机？"),
    ("ex_suffix", "HP790Ex是不是防爆对讲机？"),
    ("cqst_suffix", "HP780CQST是不是防爆对讲机？"),
    ("spaced_cqst", "HP500 CQST是不是防爆对讲机？"),
    ("chinese_suffix", "摩托罗拉 GP328D+ 防爆是不是防爆对讲机？"),
    ("cross_product", "HP780 和 HP790Ex 的输出功率与防爆认证有什么区别？"),
    ("other_category", "摄像机的名称没有Ex、CQST或防爆后缀，就能确定它不防爆吗？"),
    ("unknown_model", "你们有ZZ999CQST对讲机吗？请给出它的具体防爆认证编号。"),
)

SUITES = {"regression": REGRESSION, "holdout": HOLDOUT,
          "language_referral": LANGUAGE_REFERRAL,
          "referral_holdout": REFERRAL_HOLDOUT,
          "company_identity": COMPANY_IDENTITY,
          "radio_policy": RADIO_POLICY}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITES), default="regression")
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(execute=args.execute, audit=args.audit, cases=SUITES[args.suite])
    except Exception as error:
        # Do not expose provider exception bodies or credentials.
        print(f"Stopped: {type(error).__name__}")
        raise SystemExit(1)
