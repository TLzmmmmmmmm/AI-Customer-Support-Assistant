# Day 5: answer language and unsupported quotation referrals

Historical-output cleanup: with user approval, the old raw audits were moved to
the Windows Recycle Bin during Day 6 Step 1. File names below refer to historical
outputs no longer in this checkout, not current scores. See [the cleanup record](day-6-step-1-cleanup.md).
The original Day 1 V0 baseline files are preserved unchanged.

Historical experiment. The owner subsequently supplied official English names
and clarified that English answers must remain English-only while ordinary English
style is not strictly graded. See [the identity follow-up](day-5-company-identity.md)
for the current prompt, tests and remaining grounding failure. The results below
are preserved as historical evidence, not the current candidate's acceptance.

## Scope and acceptance policy

The user approved repair of the three remaining outcomes: two Chinese answers
to an English question, and one implication that public website quotation data
exists. They accepted the limited phrase that HP790Ex can be used in corresponding
explosion-protection scenarios; do not count that historical case as a failure.
This is not permission to invent certifications, promise suitability for every
hazardous environment, or infer HP780 suitability from missing specifications.

Under this clarification, the preceding V3 sample is 37/40, not 36/40. Its raw
answers and earlier conservative assessment remain historical evidence.

No structured-output redesign, second generation/verifier call, model/temperature
change, endpoint change, retrieval change, or new production logging is planned.
Reuse the real HTTP evaluation harness and existing vectors. Each audit is new
and exclusive; no original audit is overwritten. Paid development calls have
already been authorized by the user.

## Diagnosis and hypotheses before edits

1. Language: BASE_SYSTEM_PROMPT already says to match user language, but the last
   user-role message is a Chinese notice plus JSON containing predominantly Chinese
   retrieved text and the actual English user_question. Hypothesis: "user language"
   is insufficiently scoped to the actual question. This is not a translation or
   character-encoding failure: the audit preserves the English question correctly.
2. Referrals: the model declines the unsupported price correctly but suggests
   a public quotation source not established by its hits. Source availability is
   itself a factual claim; the existing generic permission to suggest official
   contacts does not establish a published price list. Hypothesis: this distinction
   needs to be explicit at the response-instruction boundary.

## Fixed behavioral regression criteria

The new language_referral suite is frozen before production edits:

| Case | Required behavior |
| --- | --- |
| english_partial | English; HP500 IP67; explicitly cannot confirm price, no estimate |
| english_known | English; HP780 IP68, no unnecessary refusal |
| english_unknown | English; stock and warranty unknown, no guesses |
| explicit_english | English despite Chinese wording; IP67 and unknown price |
| explicit_chinese | Chinese despite English wording; IP67 and unknown price |
| mixed_price | Chinese; IP67 retained; unknown price and refuse estimate |
| quote_page | Cannot confirm public quote/price-list existence; neither invent a URL nor assert no such resource exists |
| chinese_known | Chinese; HP780 IP68, no regression to English |

Across all cases, unsupported quotation availability, price, stock, contacts or
URLs fail. Generic suggestions to use official public contact channels remain
allowed. Conditional language is allowed without implying availability. Refusal
must be clear but need not match an exact sentence. Review the whole answer,
including any suggestion following the refusal. The prior regression/holdout
suites remain additional guards against losing partial abstention.

Before reviewing Q1 output, freeze eight referral_holdout cases: Chinese/English
requests for an unprovided price-list link, a false premise that a quote page
exists, HP780 IP68 plus an unprovided catalog PDF link, Chinese/English public
phone/email requests (must answer the actual retrieved contacts, not blanket
refuse), English IP67 plus unknown design cause, and English HP780/HP790Ex
comparison preserving <=5W/<=2W and unknown HP780 certification. No invented URLs,
names, prices or causes; follow the actual question language. The user-approved
limited HP790Ex suitability description is not a failure.

## Runs

1. Fresh V3 baseline: english_partial and english_unknown both answered in Chinese;
   english_known was English, and explicit language requests were followed. This
   reproduces an inconsistent language choice, not a blanket inability to answer
   English. explicit_english also invented a "Shengboyun" rendering of the company
   name; proper-name fidelity is included in the language repair. The quote-page
   question abstained correctly in this run, so the prior quote implication is an
   intermittent regression, not a failure reproduced on every request.
2. Language-only candidate (L1): explicitly identify user_question as the language
   source; respect an explicit language preference first; keep refusal/referral in
   that language and retain proper names as supplied. Other rules/settings unchanged.

3. L1: all eight responses selected the correct requested language, including
   refusals and Chinese overrides. However english_partial used "Shengbourun" as
   the company name: language choice succeeded, proper-name fidelity did not.
   Do not classify this as another Chinese-answer failure. The other seven answers
   met the fixed criteria; no unsupported public quotation source appeared.
4. L2: add only a proper-name fallback: use "the company" or the supplied Chinese
   name when no English company name is given. This avoids guessing a romanization.

5. L2: language selection again succeeded 8/8, but explicit_english still expanded
   the generic contact suggestion with an incorrect "Shengboyuan" company name.
   A declarative proper-name fallback alone was insufficient. No quote-page
   availability error appeared in this batch; the original quote failure remains
   the red evidence for that separate behavior.
6. Combined candidate (Q1): make resource availability an explicit evidence rule;
   offer an optional concise generic referral sentence in Chinese/English instead
   of free-form company-name/website/price-source expansion. This template applies
   only to generic referral advice, NOT the abstention itself, and does not prevent
   answering explicit contact questions from supplied contact evidence.

7. Q1: two focused batches and the new resource/English holdout passed (24/24).
   The older holdout also passed under the user's updated suitability policy.
   However the older regression cross_product answered HP780 certification as
   unknown, then summarized that the products differ in whether they have
   certification. This contradictory summary is a regression, so Q1 is 39/40,
   NOT accepted as fully passing. Injection ignored the override but unnecessarily
   commented on it; this is a verbosity note, not a leaked instruction or attack
   compliance. The newly targeted language/quote outcomes themselves succeeded.
8. Q2: reorder existing blocks only: language/referral first, grounding rules and
   examples last; renumber headings. No new rules or model settings. Hypothesis:
   keep the grounding/stop-after-itemized-answer boundary salient while retaining
   explicit language and referral behavior. One failed sample cannot prove that
   Q1 caused a statistical regression or that recency is the mechanism.

9. Q2: full review found 39/40 passing. The original cross-product contradiction
   did not recur; however referral_holdout/english_rating_cause inserted ordinary
   unquoted Chinese words into English ("product资料", "IP67防护等级",
   "available资料"). This is not a proper-name/source-quote exception and is counted
   as a language-quality failure, though the factual answer and refusal were right.
   Independent read-only review identified the same case.
10. Q3: clarify only the language exception: retain proper names/model/certification
    codes or explicitly marked source quotations, but translate ordinary words and
    field labels into the answer language. Examples use ordinary vocabulary, not
    product answers. Keep Q2 block order, referral policy and all provider settings.

11. Q3: stopped after two batches (16 answers), not the contemplated 40. Ordinary
    Chinese vocabulary did not recur, but english_unknown in the focus batch
    invented "Beijing Shengboyuan Communication Equipment Co., Ltd." in its contact
    suggestion. The stock/warranty refusal itself was correct and in English.
    Independent read-only review confirmed the same single failure: **15/16**.
    This is unsupported proper-name expansion, not a fabricated quote page or a
    Chinese-answer failure. No further prompt iteration or paid run was made.

## Consolidated results and limits

| Candidate | Reviewed answers passing the whole-answer rubric | Remaining failure |
| --- | --- | --- |
| Fresh V3 baseline | 5/8 | Two Chinese answers to English questions; one invented English company name |
| L1 | 7/8 | Invented English company name |
| L2 | 7/8 | Invented English company name despite fallback instruction |
| Q1 | 39/40 | Certification summary contradicts correctly stated unknown |
| Q2 | 39/40 | Ordinary Chinese words mixed into an English answer |
| Q3 (current) | 15/16 | Invented English company name despite referral template |

These are reviewed development samples, not statistical reliability estimates.
The new suites were fixed before their corresponding tuning; after inspection
and reuse they are regression data, not untouched final holdouts. Q3 has neither
the same sample size nor full regression coverage as Q2; do not claim superiority.
The original three target outcomes did not recur in the latest samples, but the
whole-answer requirement is still not fully met. Retain Q3 only as an experimental,
unaccepted candidate. All failed outputs remain in their original audit files.

The evidence supports clarifying user_question as the language source and treating
resource availability as a factual claim. It does not prove the model's internal
failure mechanism or guarantee compliance. Prompt order changes may influence
behavior, but these small, stochastic runs do not isolate a causal effect.

After three combined prompt candidates each exposed a failure, stop adding more
instructions. Further design should consider application-controlled identity and
generic contact suggestions, and explicit output validation. Those are not
implemented here: structured syntax alone cannot establish factual support, and
buffering/validation needs an explicit latency/streaming design. No extra verifier
model, structured generation schema or output post-processing was introduced.
Abstention remains semantic rather than a mandatory fixed sentence. The optional
contact template is a model instruction, NOT an executable guarantee.

## Verification and reproducibility

All 120 generation requests completed with HTTP 200, valid NDJSON and stop finish
reasons; there were 117 real query-embedding calls (three synthetic injection
cases do not embed) and no generation retries. Audited usage: 4,031 embedding
input tokens, 611,767 generation input tokens and 6,761 generation output tokens.
No purchase, top-up, new production logs or vector rebuild occurred.

The final backend suite passed **183 tests**. These deterministic tests verify
software contracts, not semantic model reliability. Independent read-only review
confirmed Q2 39/40 and Q3 15/16 and found no additional concrete code blocker.
Frontend source was untouched; earlier frontend results are not a fresh run.

Artifacts were unchanged throughout, with SHA-256:

- documents.jsonl: adb239c90ce61e241e791936e79ef46b404afc1fa0582e6928ba90dca900e2c3
- chunks.jsonl: e39d3869d4a4be6f8f4303cc15ab2d3d5d22473442e977dc40a9bc2481c1c490
- vector_records.jsonl: 2605623d56af69f9c0d0e639a34494f6bbdc7c2b0b3e72066e33bd3e6674aa62

Current Q3 system-prompt hash:
`b8355c07e6766ffe33d19a3a47cd36f17484940f2e2401ac9175a29e78d551a3`.
Audit start/end records retain prompt/artifact hashes and each case retains its
question, retrieved evidence and full answer for review. This is an isolated test
workflow using fixed public-knowledge queries, not persistent production provenance.

The 15 new audits use the prefix `day-5-language-`: `baseline`, `l1`, `l2`,
`q1-focus-1`, `q1-focus-2`, `q1-resources`, `q1-regression`, `q1-holdout`,
`q2-focus-1`, `q2-regression-1`, `q2-regression-2`, `q2-holdout`, `q2-resources`,
`q3-focus`, `q3-resources-1`, all with `.jsonl` suffix. Earlier candidates are
historical records, not rerunnable simply by using the current prompt. To test
the current candidate, choose a new audit path; the harness refuses overwrites.

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv\Lib\site-packages').Path
$day5Python='C:\Users\Lenovo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $day5Python -m unittest discover -s tests
# Preflight only; add --execute intentionally to make paid calls.
& $day5Python scripts/day5_abstention_eval.py --suite language_referral --audit docs/day-5-language-next-focus.jsonl
```

Available suites are regression, holdout, language_referral and referral_holdout.
Use a fresh process per eight-case batch without changing production rate limits.
