# Day 7 Step 4 — Product Existence and Commercial-Relationship Rule-Conflict Map

## Technical summary

The current prompt has the right safety intent but expresses it through several overlapping negative rules. In particular, `公司拥有产品`, `公司产品`, `公司提供产品`, and `公司业务范围` are not defined as separate predicates. That ambiguity can make the model turn a valid rule—do not infer a commercial relationship—into an invalid and unsolicited conclusion that a retrieved product is not Shengborun's product or is outside Shengborun's business scope.

The rules should converge on three primary propositions:

```text
User mention ≠ existence
Retrieved official product record → existence in published product knowledge
Existence ≠ manufacturer/agency/commercial relationship
```

Two supporting rules are still necessary: answer only the predicate the user asked about, and attribute a fact to a named source only when the supplied evidence explicitly contains that provenance.

This document is analysis only. It does not modify the production prompt or run the generation provider.

## Scope and definitions

The map covers all materially distinct clauses in `prompts.py` that govern the eight requested concepts. Repeated examples and restatements are grouped where they implement the same rule.

| Predicate | Precise meaning |
| --- | --- |
| `user_mentioned(model)` | The user typed a product or model name. This is a query input, not evidence. |
| `published_product_record(model)` | The trusted retrieval pipeline supplied a record with `type=product` for that model. |
| `exists_in_published_knowledge(model)` | The model is represented in Shengborun's curated/published product knowledge. This is the narrow existence claim established by a retrieved official product record. |
| `manufacturer(company, model)` | The company manufactures the model. |
| `supplier(company, model)` | The company supplies or sells the model. |
| `agent_or_distributor(company, brand/model)` | The company is an agent, distributor, dealer, or representative. |
| `commercially_available(model)` | The product is currently offered, in stock, or available for sale. |
| `support_scope(question)` | The question is within the AI customer's service domain. This says nothing about any commercial relationship. |
| `source_attribution(source, fact)` | A named manufacturer, website, manual, or document set is the origin of the fact. |

`公司拥有产品` should not remain a canonical predicate because it can mean knowledge-base inclusion, manufacture, intellectual-property ownership, sales availability, or representation. Those meanings require separate evidence.

## Rule-conflict map

| Rule | Intended purpose | Applicable condition | Possible conflict | Proposed canonical rule |
| --- | --- | --- | --- | --- |
| **R1 — `prompts.py:21`:** User-provided spellings cannot override the official English brand or company name. | Protect company identity and exact naming. | The answer names Shengborun Communications or the full English company name. | None for product existence, but it establishes a useful general distinction between user input and trusted facts. | User-supplied names are query data, not authoritative identity or product evidence. |
| **R2 — `prompts.py:22–24`:** Product suffixes determine explosion-proof classification only for officially named two-way radios; user-added suffixes are ignored, and a user-entered name does not establish product existence. | Apply the owner-confirmed Ex/CQST/防爆 rule without inventing products or certifications. | A two-way-radio model's official name is established and explosion-proof status is asked or required. | One sentence combines alias integrity, existence, and classification. The negative existence clause is strong, while no adjacent clause states what a retrieved product record positively establishes. | `user_mentioned(model)` does not establish existence. Once an official retrieved record establishes the formal model name, apply the suffix classification rule to that name only. |
| **R3 — `prompts.py:26–34`:** Product questions are in the assistant's duties, but being in scope does not mean the answer is known. | Separate customer-support scope from evidence sufficiency. | Any product/model/function/parameter question. | “Company product question” or “business scope” can be misread as a product-ownership test. A question can be in support scope even when a commercial relation is unknown. | `support_scope(question)` is independent of product existence and every commercial relationship. |
| **R4 — `prompts.py:39–53`:** Prioritize company-product topics, but do not infer that the company provides a product/service/solution merely because the user mentions it. | Prevent a user assertion from becoming a company fact. | The only evidence for the product/category/solution is the user's wording. | Correct when only the user mentions the item; overbroad if applied after a trusted product record is retrieved. It also uses `提供` without distinguishing “published information” from “sells/supplies.” | User mention alone does not prove existence or provision. A retrieved official product record supersedes the *absence of existence evidence* only; it does not establish supply or sale. |
| **R5 — `prompts.py:58–62`:** Application-provided and system-retrieved product/company knowledge is an allowed factual basis. | Define trusted evidence sources. | The application supplies retrieved knowledge for the current request. | It calls the records “产品…或公司知识” but does not define the exact claim a `product` record supports. That leaves a semantic gap between R4's prohibition and answering product facts. | A trusted retrieved record may support only the claims contained in its text plus its record-class implication: `type=product` establishes inclusion/existence in published product knowledge. |
| **R6 — `prompts.py:64–83`:** Model knowledge, product-category inference, and model-name inference are not company-fact evidence. | Stop latent knowledge and naming patterns from creating company facts. | A fact is absent from trusted supplied evidence. | “根据型号名称进行的推断” could be applied too broadly to reject a model even when retrieval found its official record. | Do not infer facts from a name alone. Facts explicitly stated in the matched product record, and only those facts, remain usable. |
| **R7 — `prompts.py:86–107`:** Without reliable support, do not claim the company owns a category, series, model, product line, solution, function, parameter, or use case. | Prevent unsupported portfolio and capability claims. | The answer would assert a Shengborun-specific fact not established by evidence. | `拥有` and `公司产品` conflate catalog inclusion, manufacture, supply, agency, and ownership. Repeated one-sided negative checks can induce the converse assertion—“不是盛博润的产品”—even though lack of relationship evidence proves neither side. | Remove `公司拥有产品` as an umbrella concept. Evaluate `exists_in_published_knowledge`, `manufacturer`, `supplier`, `agent_or_distributor`, and `commercially_available` independently. Never infer either a positive or negative relationship from missing evidence. |
| **R8 — `prompts.py:109–119`:** General industry features cannot be described as features of “our products”; general knowledge must be labeled separately. | Prevent industry defaults from becoming company-specific product claims. | The user asks for general knowledge or the retrieved record omits a feature. | “Our product” remains semantically ambiguous, but this clause mainly governs feature attribution rather than existence. | A feature belongs to a product only when that product's trusted evidence supports it. This does not decide manufacturer, supply, or agency status. |
| **R9 — `prompts.py:122–162`:** Product, commercial, inventory, pricing, delivery, and service facts require reliable evidence and cannot be guessed. | Prevent unsupported specifications and commercial commitments. | Any requested product or commercial fact. | Product identity/existence and commercial availability appear in one broad rule family, encouraging them to be treated as one fact. | Product-record existence and commercial availability are separate. A record can establish the former while price, stock, sale, supply, or delivery remain unknown. |
| **R10 — `prompts.py:165–180`:** A recommendation requires the specific model to exist and its relevant properties and fit to be supported. | Ground product selection. | The assistant recommends a specific product. | The clause requires existence but never defines which evidence is sufficient to establish it. | For recommendation eligibility, a matched official `type=product` record establishes published-knowledge existence; suitability still requires explicit product-property and scenario evidence. |
| **R11 — `prompts.py:202–224`:** Conversation references may resolve a model, but history and prior assistant statements are not factual sources. | Preserve conversational continuity without laundering hallucinations into facts. | Follow-up turns with pronouns or previously named products. | A user-provided or prior-assistant model name can resolve the query target but must not be mistaken for existence evidence. | Conversation history may identify the intended entity; existence still requires trusted system/retrieval evidence. |
| **R12 — `prompts.py:361–381`:** Answer only what was asked, avoid unsolicited expansion, and stop after a complete direct answer. | Keep answers concise and prevent unsupported tangents. | Every response. | This broad positive behavior competes with numerous stronger, specific negative company-ownership checks. The baseline-007 output followed the negative concern instead of stopping after the requested facts. | Determine the requested predicate(s), answer those only, and do not introduce product ownership, manufacture, supply, agency, distribution, or business-scope conclusions unless asked. |
| **R13 — `prompts.py:384–399`:** Before output, check whether the answer implies that the company owns an unconfirmed product/model/series/service/solution; otherwise omit it and say it cannot be confirmed. | Final guard against unsupported company claims. | Before emitting any Shengborun-related fact. | This is the clearest conflict source. The ambiguous and one-directional “owns” check can trigger an unsolicited negative claim or refusal even when the user asked only for retrieved product facts. It also instructs an abstention without checking whether that unknown predicate was requested. | Replace the ownership check with predicate-specific checks. If an unasked relationship lacks evidence, omit it silently; if the user explicitly asked it, say it cannot be confirmed. |
| **R14 — `prompts.py:424–429`:** No evidence means no company fact; insufficient product data means no recommendation or inference. | Provide a concise safety priority. | Conflicting goals or missing evidence. | Safe in isolation, but “company fact” remains undefined when an official product record is retrieved. | Evidence supports claims at the narrowest level it actually establishes; do not broaden product-record facts into relationship claims. |
| **R15 — `prompts.py:446–454`:** Retrieved company data is the required factual basis; each model and attribute must be supported independently; absence is generally not proof of nonexistence. | Establish the RAG evidence boundary and localize unknowns. | A company-specific RAG answer. | This section says retrieved data is authoritative for facts but still does not explicitly say that a retrieved product record establishes published-knowledge existence. R7/R13 can therefore dominate it. | A valid `type=product` record establishes that the model exists in the curated/published product knowledge. Its text supports stated attributes only; omitted attributes remain governed by the project's chosen closed/open-world feature policy. |
| **R16 — `prompts.py:459`:** Do not voluntarily introduce supply, agency, distribution, representation, ownership, or other commercial relationships, including statements that they cannot be confirmed. | Prevent baseline-007-style tangents. | The user did not ask about a commercial relationship. | Correct but late and isolated. It competes with R13's instruction to turn any unconfirmed implied ownership into an explicit “cannot confirm” statement. | Unasked commercial predicates are out of answer scope and must be omitted, not evaluated aloud. |
| **R17 — `prompts.py:460`:** Do not attribute facts to a manufacturer, official website, manual, document set, or source unless the supplied evidence explicitly establishes provenance. | Prevent fabricated source attribution. | The answer names the origin of retrieved facts. | The production model context currently omits `source_url`, `source_files`, `chunk_id`, and `parent_document_id`. A product brand or title is not provenance, so most named-source attribution is unsupported even when backend provenance exists. | Name a source only when provenance is included explicitly in the model-visible evidence. Otherwise say “根据现有资料” or omit source wording. |
| **R18 — `prompts.py:461`:** If the user explicitly asks about a commercial relationship and evidence does not establish it, clearly say it cannot be confirmed. | Preserve correct open-world abstention for supplier/agency/ownership questions. | The user explicitly asks whether Shengborun manufactures, supplies, sells, represents, distributes, or owns a product/brand. | No logical conflict with R16, but the pair depends on reliable detection of what the user actually asked. It also needs symmetric treatment: missing evidence establishes neither “yes” nor “no.” | For an explicitly requested commercial predicate, answer from explicit evidence; otherwise say it cannot be confirmed. Do not infer a negative relationship from silence. |

## Current model-visible retrieval evidence

The backend preserves full `RetrievalResult` provenance, but `rag_context.build_retrieved_context()` currently sends only:

```json
{
  "type": "product",
  "section": "...",
  "text": "..."
}
```

The generated provider message does not contain `source_url`, `source_files`, `chunk_id`, or `parent_document_id`. Therefore:

- `type=product` plus the system's statement that current retrieved company knowledge is trusted can establish **existence in the curated/published product knowledge**.
- Product text can establish the product facts explicitly written in that text.
- A brand/model string does not establish who manufactured the product.
- The record does not establish present supply, sale, stock, agency, distribution, representation, or ownership unless its text explicitly says so.
- The model cannot accurately name a website/manual/document origin from hidden backend provenance. Backend provenance preservation is not the same as model-visible provenance.

## Conflict diagnosis

The prompt currently has an asymmetric structure:

```text
Many repeated rules say:
do not infer that Shengborun owns/offers/has a product

But no equally explicit rule says:
a trusted retrieved product record establishes a narrow form of product existence

Resulting failure mode:
retrieved product facts are answerable
→ model notices an unconfirmed “ownership” concern
→ model volunteers a negative commercial/business-scope conclusion
→ model may fabricate a source attribution to explain the inconsistency
```

This is not a retrieval failure. It is a predicate-definition and rule-priority conflict in the generation policy.

## Proposed canonical rule set

The following is a semantic target for a later prompt revision, not a prompt change made in this step.

### Canonical rule 1 — User mention is not evidence

```text
A product/model name appearing only in the user's message does not prove that
the product exists, that it appears in Shengborun's published product knowledge,
or that Shengborun has any commercial relationship with it.
```

### Canonical rule 2 — A retrieved official product record establishes narrow existence

```text
When the trusted retrieved context contains a valid product record for a model,
you may treat that model as existing in Shengborun's published product knowledge
and answer the facts explicitly supported by that record.
```

“Published product knowledge” is deliberately narrower and less ambiguous than “盛博润的产品.”

### Canonical rule 3 — Existence is independent of commercial relationships

```text
Existence in published product knowledge does not by itself establish who
manufactures the product or whether Shengborun supplies, sells, owns, represents,
acts as agent/distributor for, or currently has stock of it. Each relationship
requires its own explicit evidence. Missing evidence proves neither the positive
nor the negative form of the relationship.
```

### Canonical rule 4 — Answer only requested predicates

```text
Do not introduce manufacturer, supplier, agency, distribution, representation,
ownership, availability, or business-scope claims unless the user asked about
that relationship or it is strictly necessary to answer the question.
```

### Canonical rule 5 — Explicit commercial questions remain open-world

```text
If the user explicitly asks about one of those commercial relationships, answer
only when the evidence explicitly establishes it; otherwise say it cannot be
confirmed. Do not convert absence of evidence into a negative relationship.
```

### Canonical rule 6 — Source attribution requires model-visible provenance

```text
Attribute a fact to a manufacturer, website, manual, or documentation set only
when that provenance appears explicitly in the evidence supplied to the model.
Otherwise use neutral wording such as “根据现有资料” or omit attribution.
```

## Baseline-007 application

Question:

```text
海能达 PNE380 是什么设备？它的便携性、待机时间和组网能力有哪些公开参数？
```

Correct rule application:

1. The user mentions PNE380; this alone would not establish existence.
2. Retrieval supplies a trusted `type=product` PNE380 record; this establishes the product's presence in published product knowledge.
3. The record text supports the requested device type, size/weight, standby time, throughput, node count, and networking facts.
4. The user did not ask who manufactures, supplies, represents, distributes, or owns it, so those predicates must not appear.
5. The model-visible context does not include a source URL or named document provenance, so it must not claim the facts came from Hytera product documentation, Shengborun's website, or user-provided materials.
6. The answer should stop after the supported requested facts.

## Consolidation recommendation

During a later authorized prompt revision:

- Replace repeated ambiguous phrases such as `公司拥有某个具体型号` with predicate-specific rules.
- Keep the user-input boundary, but state its condition explicitly: **when the only evidence is the user's message**.
- Add one positive rule defining exactly what a valid retrieved `type=product` record establishes.
- Keep commercial relationships open-world and independent from published-knowledge existence.
- Make “answer only what was asked” the routing rule that decides whether an unknown should be stated or silently omitted.
- Retain source-attribution protection, aligned with the fields actually visible to the model.
- Remove or rewrite R13's blanket instruction to convert every unconfirmed implied ownership fact into an explicit abstention; abstention is required only when that predicate was asked.

The goal is deletion and consolidation, not another layer of negative clauses.

## Methodology and evidence

- Inspected the complete production system prompt in `prompts.py`, including identity, scope, evidence, company-fact, product/commercial, recommendation, style, pre-answer checks, and RAG trust-boundary sections.
- Searched the prompt for every material occurrence of product/model, business scope, ownership/provision, manufacturer, supply, agency/distribution, brand, source attribution, user-provided names, and retrieved evidence.
- Inspected `rag_context.py` and `build_rag_messages()` to distinguish backend-preserved provenance from fields actually supplied to the model.
- Applied the rules to the owner-reviewed baseline-007 failure without making a new provider call.

## Limitations and robustness

- “Official product record” is an architectural interpretation of a trusted, pipeline-produced `type=product` record. The current serialized item does not itself contain an explicit `official=true` marker.
- This map does not decide whether model-visible provenance should later be expanded. The current request is about prompt-rule conflicts, and production logging/privacy constraints remain unchanged.
- The analysis predicts why the present clauses can produce baseline-007 behavior; it does not claim a canonical rewrite is effective until a later TDD and targeted provider regression validates it.
- No historical evaluation artifacts were altered.

## Recommended next step

After owner review, revise the prompt by consolidating—not appending—the conflicting rules into the six canonical rules above. Then use offline prompt-structure tests followed by the already approved small seen-regression set; do not run the unseen V1.1 holdout until that gate passes.

## Further questions for owner review

1. Approve `存在于盛博润已发布的产品知识中` as the canonical narrow meaning of a retrieved official product record, instead of the ambiguous `是盛博润的产品`.
2. Confirm that product supply/sale remains open-world unless explicit evidence states it, even when a product page is present in the knowledge base.
3. Decide separately whether future model-visible context should carry a neutral source label; this is not required to solve the current unsolicited-attribution failure if the assistant simply omits unsupported attribution.
