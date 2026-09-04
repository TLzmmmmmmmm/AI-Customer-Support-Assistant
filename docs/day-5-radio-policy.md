# Day 5: owner-confirmed radio classification and concise answers

Historical-output cleanup: with user approval, the old raw audits were moved to
the Windows Recycle Bin during Day 6 Step 1. File names below refer to historical
outputs no longer in this checkout, not current scores. See [the cleanup record](day-6-step-1-cleanup.md).
The original Day 1 V0 baseline files are preserved unchanged. The later user decision
allows accurate consultation suggestions and repeated summaries; strict scope scores
below are historical, not Day 6 failures.

## Authoritative requirement

The company owner explicitly confirms this classification for all walkie-talkie
products in this project: official product names ending in Ex, CQST or 防爆 are
explosion-protected; all other walkie-talkies are non-explosion-protected.
This is a company-supplied rule, not a universal standard inferred by the model.
It does not apply to cameras, antennas, accessories or other product categories.
Use the actual official name (including spaced suffixes), not a suffix invented
by the end user. A fabricated model name does not establish catalog availability.
Specific certification codes, protection levels and design causes still require
product evidence; the suffix alone cannot supply those details.

The owner also corrected the answer scope: a question comparing power and
explosion protection requires only those requested attributes. Do not append
unsolicited explanations about power upper bounds, actual power, design causes,
selection advice, follow-up topics or contact suggestions. Keep <= in the actual
values. Explain upper bounds or causes only when the user actually asks about
them; a requested but unsupported cause must be explicitly unconfirmed.

## Fixed expectations before the production change

| Case | Expected |
| --- | --- |
| plain_radio | HP780 non-explosion-protected, not unknown |
| ex_suffix | HP790Ex explosion-protected |
| cqst_suffix | HP780CQST explosion-protected |
| spaced_cqst | HP500 CQST explosion-protected |
| chinese_suffix | GP328D+ 防爆 explosion-protected |
| cross_product | HP780 <=5W/non-explosion-protected; HP790Ex <=2W/its recorded protection levels; stop after comparison |
| other_category | Do not classify cameras by the radio naming rule |
| unknown_model | Cannot confirm ZZ999CQST availability or certification; invent neither |

For simple yes/no classification, one direct sentence suffices. Relevant evidence
may be stated briefly; unrequested specifications or speculative commentary fail
the scope requirement. The comparison must preserve complete bounds/units and
provide no extra paragraph explaining the bounds or hypothesizing design intent.

Updated expectations for existing suites:

- regression/omission_pressure: HP780 is non-explosion-protected because of the
  owner-confirmed rule, not because missing documentation universally means absence.
- regression/cross_product and referral_holdout/english_comparison: use the
  classifications and recorded parameters above, without unsolicited explanation.
- holdout/other_comparison: HP710Ex protection details from evidence; HP500 is
  non-explosion-protected, not unknown.
- holdout/inference_pressure: HP780 is non-explosion-protected; do not recommend
  it for an explosion-protected use case. Do not promise universal suitability of
  the other model or invent certification details.
- holdout/power_qualifier: the question explicitly asks about constant power, so
  explaining <= is required here, not forbidden.
- regression/causal_pressure and holdout/known_and_cause: explicitly cannot confirm
  the requested design cause; do not hypothesize even with a disclaimer.
- All other unknown facts, identity, English-only, grounding, anti-injection and
  privacy requirements remain unchanged. Older scores are historical, not rewritten.

## Verification plan

Run the frozen radio_policy suite before and after the prompt change, then rerun
regression, holdout and company_identity. Real APIs are already authorized; keep
each batch at eight cases and create new exclusive local audits. Synthetic injection
uses the existing fixture with real generation. No production logging, vector
rebuild, retrieval change, output post-processing or additional generation stage.

## Experiment record

The baseline correctly used existing explicit positive-protection evidence but
failed the new HP780 classification and concise comparison requirements. Its
cross_product answer called HP780 unknown and appended an unrequested upper-bound
warning. cqst_suffix also added an unrequested product-line-ownership refusal.
These are preserved in day-5-radio-policy-baseline.jsonl. Historical absence-based
refusals predate the owner's new rule and are not retrospectively hallucinations.

R1 adds the rule as trusted company information and replaces the conflicting
blanket prohibition on name-based classification. Fictional examples now respect
the rule. Both R1 comparison responses (radio_policy and regression) give the
requested classifications and parameters without the rejected final paragraph.
However multiple R1 answers still append unrequested contact suggestions or other
attributes. Existing sections 8 and 15 explicitly allowed automatic referrals,
conflicting with the new concise-only scope. R1 is not fully accepted on scope.

R2 removes those referral permissions: provide contact guidance only when the user
asks for consultation, contact details or next steps. Direct classification gets
the classification only; brand-only questions get the brand only. Do not change
facts, retrieval or provider settings. Rerun the same four final suites using new
r2 audit filenames; preserve R1 outputs, including remaining scope violations.

## Final results and remaining limits

| Candidate | Factual/classification/identity checks | Whole-answer scope checks |
| --- | --- | --- |
| R1 (32 answers) | 32/32 | 22/32 |
| R2 (32 answers, current) | 32/32 | 22/32 |

Independent read-only review confirmed both sets. The figures are development
sample reviews, not reliability estimates. R2 has no measured aggregate scope
improvement over R1 in these batches; do not infer effectiveness just because
conflicting referral permissions were removed. Keep the user-confirmed rule and
coherent prompt policy, but do not declare the concise-answer requirement solved.

R2 classification checks include Ex, CQST, spaced CQST, Chinese 防爆 and plain
names, rejecting application to cameras and refusing to invent an unknown model.
Name spelling and English-only answers passed in the reviewed samples. The two
original comparison answers both correctly classify HP780/HP790Ex and preserve
power limits/recorded protection levels. Only one of those two strictly stops
after the requested comparison; the regression answer adds:

> 两款在输出功率和防爆认证上有明显区别。

That redundant final sentence fails the user's scope requirement even though it
does not invent a design cause. The earlier rejected paragraph about actual power
and a possible design cause did not recur in these two answers. Explaining power
bounds when explicitly asked (holdout/power_qualifier) still works.

Ten clear R2 scope failures, with audit names prefixed day-5-radio-policy-r2-:

- confirmed/other_category: unsolicited contact suggestion.
- confirmed/unknown_model: unsolicited typo question and other-model search offer.
- regression/cross_product: unrequested summary after the comparison.
- regression/mixed_price and mixed_unknown: unsolicited contact suggestions.
- holdout/english_partial and all_unknown: unsolicited contact suggestions.
- identity/full_name_en: adds unrequested brand to the full-name answer.
- identity/english_unknown and brand_quote_link: unsolicited contact suggestions.

holdout/other_comparison explains the naming-rule basis after the user specifically
warns not to infer absence from missing documentation. It is somewhat repetitive
but directly responsive; not counted as an additional scope failure. Brief relevant
evidence is not failed merely for being more than one sentence.

All 72 requests across baseline, R1 and R2 completed with HTTP200, valid NDJSON and
stop finishes: 72 real generations and 70 real query embeddings; two injection
fixtures used synthetic retrieval and real generation. Each audit records unchanged
knowledge/vector hashes, with no generation retries. The final backend suite passed
183 tests; git diff --check passed. No production logging, retrieval, model settings,
stream protocol, knowledge artifacts or frontend source was changed. No deployment
or commit was performed. New docs/audits remain local under the ignored docs folder
unless explicitly included in a later commit.

Stop after R2 with all failures preserved. The classification correction is
implemented and verified in these samples; overall Day5 acceptance remains
incomplete because model output still sometimes exceeds the requested scope.
No post-processing, hidden retry-until-pass or additional review model was added.
