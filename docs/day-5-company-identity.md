# Day 5: owner-confirmed English identity and language policy

Historical-output cleanup: with user approval, the old raw audits were moved to
the Windows Recycle Bin during Day 6 Step 1. File names below refer to historical
outputs no longer in this checkout, not current scores. See [the cleanup record](day-6-step-1-cleanup.md).
The original Day 1 V0 baseline files are preserved unchanged.

Historical experiment. The current [radio-rule follow-up](day-5-radio-policy.md)
adds an owner-confirmed classification rule and stricter answer-scope requirements.
The names below remain authoritative; the run results below describe the earlier
prompt, not the current candidate.

## Approved change

The company owner supplied these authoritative names:

- Brand: **Shengborun Communications**
- Full English Name: **Beijing Shengborun Communication Equipment Co., Ltd.**

These are system-level identity facts, not guesses or names inferred from chunks.
Keep the two names distinct and use their exact spelling. User-supplied misspellings
must not override them. No knowledge artifacts or embeddings are changed.

English questions require entirely English answers, including refusals, referrals
and quotations. An explicit requested response language takes precedence (the
existing English-question/Chinese-answer override test remains valid). Ordinary
English wording, translation choices and style are not graded for polish, provided
they preserve facts, parameters and the meaning of abstention. Chinese mixed into
English remains a failure. Clear abstention and no speculation remain mandatory.

## Fixed checks, before production edits

The company_identity suite contains eight cases: English brand, English full name,
both names requested in Chinese, correction of an incorrect English name, unknown
HP780 inventory/warranty, full name plus retrieved public phone/email, HP500 IP67
plus unknown design cause, and brand plus unprovided price-list URL.

Names must match the owner's spelling wherever used; explicit name questions must
answer the available identity instead of refusing. English answers must not contain
Chinese text. Phone/email must agree with retrieved evidence. Missing stock,
warranty, design cause and price-list availability must remain unconfirmed, without
invented values, causes or links. Model/units/certification fidelity is unchanged.
The prior language_referral and regression suites will also be rerun. Reviews are
evidence-based manual semantic assessment, not an LLM judge or prompt-substring test.

## Baseline

The first sandboxed attempt stopped at an embedding error (HTTP 503), before any
generation. It is preserved in day-5-company-identity-baseline.jsonl and is not a
semantic sample. A separately approved unrestricted rerun is preserved in
day-5-company-identity-baseline-network.jsonl.

The eight-case rerun completed successfully at the transport layer. All six
explicit identity questions lacked the newly required identity answers. The
inventory/warranty case invented "Beijing ShengbogRun communication equipment
Co., Ltd."; the incorrect-name and contact cases also mixed Chinese company names
into English. Only english_rating_cause passed the entire updated rubric: **1/8**.
The identity refusals were justified under the old missing-name context; this is a
new authoritative-information requirement, not evidence that refusal was wrong then.

## Implementation

Add the official names to the system identity section; explicitly permit them as
trusted facts even when retrieval does not repeat them. Replace the older permissive
English-answer rule and Chinese-company-name fallback with English-only behavior.
Allow ordinary English phrasing freedom without relaxing evidence or abstention.
Keep optional referral guidance, retrieval, streaming and logging unchanged.

## Results

| Final-candidate suite | Whole-answer pass | Notes |
| --- | --- | --- |
| company_identity | 8/8 | Correct names; unknown facts still withheld |
| language_referral | 8/8 | All requested-English answers entirely English; explicit Chinese override retained |
| regression | 7/8 | Cross-product answer speculates about a design cause |
| Total | **23/24** | Bounded identity/language change verified in these samples; overall Day 5 not fully accepted |

Independent review confirmed the identity suite 8/8 and baseline 1/8. The remaining
16 answers were also reviewed against their evidence. In
day-5-company-grounding-regression.jsonl, cross_product first states the power
limits and unknown HP780 certification correctly, then adds:

> HP790Ex 作为防爆机型，其较低的功率标注可能与防爆设计有关，但现有资料未说明这一因果关系，无法确认。

That disclaimer does not undo the unsupported speculation; this answer fails the
unchanged no-guessing requirement. The user-approved limited suitability wording
is not permission to infer a design cause. Stop this bounded name/language change
without claiming that the older grounding issue is solved or adding unrelated
generation infrastructure. Do not rerun until a pass and discard this failure.

The 183-test backend suite passed after the production edit. All 24 final answers
had successful HTTP/NDJSON transport and stop finishes; artifact hashes were
unchanged. Including the eight-case baseline, 32 generation requests and 31
successful embedding queries ran, plus the separately recorded failed sandbox
embedding attempt (no generation). No model-based judge or extra production model
call was added. The one injection case uses synthetic retrieval with real generation;
the other cases use the real embedding provider, Retriever and existing vectors.

Audits are exclusively created local evaluation files, not production logs:

- day-5-company-identity-baseline.jsonl: failed sandbox attempt
- day-5-company-identity-baseline-network.jsonl: eight-case pre-edit baseline
- day-5-company-identity-confirmed.jsonl: final identity answers
- day-5-company-language-regression.jsonl: final language/referral answers
- day-5-company-grounding-regression.jsonl: final grounding answers including failure

Historical audits and scores are not overwritten or retrospectively declared
passing under the newly supplied identity. The passing cases are finite development
samples, not a guarantee of future model compliance. No deployment or commit was
performed for this change. The docs directory is ignored by default in this repo;
these new local evaluation files require explicit inclusion if later committed.

To reproduce, use the existing live evaluation runner with --suite company_identity
(or language_referral / regression), a new --audit path, and --execute only when
intentionally making paid calls. Omitting --execute is a configuration preflight.
