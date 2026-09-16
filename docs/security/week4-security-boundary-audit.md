# Week 4 Security / Red-Team — Step 1

# AI Customer-Support Security Boundary Audit

**Audit date:** 2026-09-16  
**Scope:** Current production path for `POST /api/chat-stream`  
**Change policy:** Documentation-only audit; no production behavior or tests changed.

## 1. Executive summary

The production request path is:

```text
POST /api/chat-stream
→ FastAPI/Pydantic request validation
→ per-client rate limit
→ process-local LLM concurrency limit
→ GraphRouteOrchestrator.run()
→ LangGraph route node
→ deterministic route, retrieval route, or agentic route
→ registered ToolExecutor when a tool is required
→ buffered model completion
→ render_answer()
→ one answer delta, optional trusted citations, done
```

The strongest current controls are structural and deterministic: request shape and size validation, role-sequence validation, strict router-output parsing, a fixed tool allowlist, strict Pydantic tool-argument schemas, a three-call agent budget, read-only registered tools, safe error contracts, and answer/source sanitization.

The main semantic controls remain LLM-dependent. The system prompt says that user text, history, retrieved text, and tool observations are data rather than instructions, and it prohibits prompt disclosure, unsupported facts, and unsupported commercial claims. The application does not, however, perform a deterministic semantic policy check on the final answer. Consequently, S1, S2, S5, S6, S7, and S10 are not universally guaranteed by code even where prompt-contract tests or sampled live evaluations have passed.

## 2. Trust inputs and ownership

Inputs treated as untrusted at runtime are:

- the complete HTTP body, including every `user` message and every client-supplied `assistant` history message;
- request metadata used by controls, especially `request.client.host` for rate limiting;
- LLM router completions, generated answer text, and model-proposed tool calls/arguments;
- embedding-provider responses and failures;
- retrieved chunk text and metadata at the prompt boundary, even though the deployed corpus is locally curated;
- tool observations at the prompt boundary;
- model/tool/retrieval exception details;
- model-generated URLs, reference sections, and citation claims;
- source objects passed to final rendering.

Security-boundary ownership is intentionally split:

- FastAPI/Pydantic owns request shape, size, and conversation-role validation.
- `rate_limit.py` and `concurrency.py` own admission controls.
- `HybridRouter` and LangGraph own capability selection and execution flow.
- `ExactEntityResolver` owns exact product-identity matching; it does not establish arbitrary facts from user text.
- prompt builders own the data/control framing presented to the model.
- `ToolExecutor`, `TOOL_SPECS`, and the immutable registry own tool authorization and argument validation.
- deterministic tool implementations own authoritative lookup constraints and safe tool errors.
- `render_answer()` and citation sanitization own URL/reference removal and trusted-source rendering, not general semantic policy enforcement.
- FastAPI exception handlers own public error redaction.

## 3. Architecture boundary table

| Boundary | Untrusted input | Enforcement | Expected failure mode |
| --- | --- | --- | --- |
| HTTP request | JSON body, message roles/content, client-supplied history | `models.ChatMessage`, `models.ChatRequest`; non-empty trimmed content; maximum 4,000 characters per message, 20 messages, 20,000 total characters; strict alternating `user`/`assistant` sequence starting and ending with `user` | HTTP 422 with `validation_error`, generic Chinese message, and request ID; orchestration does not start |
| Rate limit | Request frequency and client address | `rate_limit.enforce_rate_limit`; in-memory sliding window keyed by `request.client.host`, currently 10 requests per 60 seconds | HTTP 429 with `rate_limit`, safe message, request ID, and `Retry-After` |
| Concurrency limit | Number of admitted LLM request lifecycles | `concurrency.try_acquire_llm_slot`; process-local bounded semaphore, currently five slots | HTTP 503 with `concurrency_limit`; no orchestration; slot ownership/release is handled exactly once after acquisition |
| Router | Latest query, history, exact-entity matches, LLM classifier output | `routing.router.HybridRouter.route`; deterministic patterns and exact entity lookup first; strict single-label parsing for LLM fallback; classifier has a 16-token cap | Deterministic unsupported inventory request routes to fixed fallback; malformed/invalid classifier output routes to fixed fallback with `ROUTING` failure; provider errors propagate to safe HTTP mapping |
| Exact entity resolver | Product/model strings embedded in user text | `ExactEntityResolver.resolve` and `resolve_canonical_identifier`; NFKC/case/separator normalization, ASCII alphanumeric boundaries, exact catalog aliases, collision checks | No match or an explicit ambiguous/unknown tool result; user text alone does not create a catalog entity |
| Retriever | Raw latest user query; embedding response; local vector records | `Retriever.retrieve`; non-blank query and positive `top_k`; one query vector required; typed/validated records and results; exact-entity hits are combined with dense retrieval | Invalid local call raises validation error; retrieval/provider failures are classified and become safe HTTP 503 responses on the production knowledge path |
| Prompt assembly | User/history text, retrieved text, tool observation text | `build_direct_messages`, `build_rag_messages`, `build_tool_messages`, and `build_agent_messages`; application system messages precede data; RAG/tool payloads are deterministically JSON serialized and labeled as data | Structural delimiter-breaking is prevented by JSON serialization, but semantic obedience to the data/control instruction remains model-dependent |
| Agent | LLM completion and proposed tool-call batch | `agent_step_node`, `normalize_agent_turn`, `tool_call_batch_rejection`; normalized function calls only, unique call IDs, maximum three processed calls, whole oversized batch rejected before execution | Malformed completion, duplicate IDs, or oversized/exhausted calls terminate with `SAFE_AGENT_ANSWER`; invalid calls receive failed traces and do not reach a tool |
| Tool executor | Model-proposed tool name and JSON arguments; deterministic route arguments | `ToolExecutor.execute` / `execute_named`; `TOOL_SPECS` allowlist; JSON object requirement; strict Pydantic models with `extra="forbid"`; registry lookup after validation | Safe `INVALID_ARGUMENT` observation; unknown tool name is redacted as `tool_executor`; callable is not invoked |
| Tool implementation | Validated product ID/query and local dependency results | `DeterministicTools`; exact canonical IDs, published inventory only, fixed product-only search controls, bounded query/product IDs, typed result model | Safe structured domain error (`PRODUCT_NOT_FOUND`, `AMBIGUOUS_PRODUCT`, `TOOL_UNAVAILABLE`, or `TOOL_EXECUTION_ERROR`); unexpected details are redacted |
| Answer rendering | Raw model answer and proposed source objects | `citation.render_answer`, `sanitize_generated_answer`, `collect_sources`; strips raw URLs and model-written reference sections; validates/deduplicates HTTP(S) source objects; citations come only from backend sources | Sanitized answer is emitted; invalid sources are dropped; fully removed/empty answer becomes `SAFE_AGENT_ANSWER`; no general semantic claim checker is present |
| Public error boundary | Exceptions and internal/provider error text | `validation_exception_handler`, `http_exception_handler`, `unhandled_exception_handler`; allowlisted public codes/messages and error type only in telemetry | Safe JSON error containing code, generic message, and request ID; no traceback or internal exception message in the response |

## 4. Security invariant audit

The classifications below use:

- **Deterministic:** enforced by application code independently of model cooperation.
- **Partial:** a deterministic sub-boundary exists, but the full invariant still depends on model behavior.
- **LLM-dependent:** expressed primarily in prompts/evaluation; no deterministic final-answer enforcement exists.

### S1. User input cannot override system/policy instructions

- **Relevant production code:** `prompts.BASE_SYSTEM_PROMPT`, `SYSTEM_PROMPT`, `RAG_SYSTEM_INSTRUCTIONS`; `build_direct_messages()`, `build_rag_messages()`, `build_tool_messages()`; `agent.runtime.AGENT_TOOL_POLICY` and `build_agent_messages()`.
- **Enforcement layer:** Prompt assembly and model instruction hierarchy.
- **Classification:** **LLM-dependent**, with deterministic message ordering and JSON data envelopes.
- **Current failure behavior:** There is no semantic postcondition that detects policy override. A compliant model ignores the attack; a noncompliant answer can pass through unless URL/reference sanitization happens to alter it.
- **Existing automated coverage:** `tests/test_prompts.py` checks instruction presence, system-message position, data envelopes, history preservation, and JSON round trips. `tests/test_day7_production_smoke.py` checks one prompt-injection answer contract, while `eval/results/day7-week3-production-smoke-v1.jsonl` records a prior sampled production-smoke pass.
- **Obvious untested gap:** No broad production red-team matrix covers direct, RAG, tool, agentic, multilingual, encoded/obfuscated, split-across-history, or multi-turn override attacks. A sampled pass is not a deterministic guarantee.

### S2. System prompt, hidden policy and internal configuration must not be directly disclosed

- **Relevant production code:** secrecy instruction in `prompts.BASE_SYSTEM_PROMPT`; `error_handling.py`; safe tool errors in `agent.executor` and `support_tools.service`; `citation.render_answer()`.
- **Enforcement layer:** Model policy for generated disclosure; deterministic error and tool-error redaction for exception-driven disclosure.
- **Classification:** **Partial**. Error-path disclosure is deterministically constrained; model-generated prompt/config disclosure is LLM-dependent.
- **Current failure behavior:** Exceptions produce generic public errors and logs record bounded fields/error types. There is no output filter for system-prompt text, configuration values, or hidden-policy fragments, so model disclosure could be emitted as ordinary answer text.
- **Existing automated coverage:** `tests/test_error_handling.py` verifies public error and telemetry redaction; `tests/test_agent_executor.py` and `tests/test_support_tools.py` verify internal tool details are redacted; `tests/test_day7_production_smoke.py::test_prompt_injection_prompt_leakage_still_fails` validates the smoke evaluator for a few literal leakage markers.
- **Obvious untested gap:** No production test attempts partial, paraphrased, translated, encoded, or history-induced extraction of system prompts/configuration. The renderer does not enforce this invariant.

### S3. User text cannot directly trigger arbitrary tool execution

- **Relevant production code:** `routing.deterministic.execute_deterministic_route()`; `agent_graph.nodes.agent_step_node()` and `execute_tool_node()`; `agent.models.TOOL_SPECS`; `agent.executor.ToolExecutor`.
- **Enforcement layer:** Router/LangGraph execution flow plus tool executor.
- **Classification:** **Deterministic** for arbitrary/unregistered capabilities. User text may influence route selection or a model proposal, but it is never interpreted as an executable tool name/argument payload by the HTTP layer.
- **Current failure behavior:** Deterministic routes map only to fixed tool names. Agentic proposals must survive model-call normalization, batch checks, allowlist checks, and schema validation. Unknown names become a safe failed observation and no registry callable runs.
- **Existing automated coverage:** `tests/test_agent_executor.py::test_unknown_tool_is_redacted_and_never_executes_registry`, invalid-shape/argument tests, `tests/test_agent_graph.py` batch-budget tests, deterministic-routing tests, and the `dev_fallback_04` evaluation case for a requested `delete_customer_database` operation.
- **Obvious untested gap:** Add an end-to-end HTTP case that embeds fake function-call syntax and an unregistered destructive tool name in user text and asserts zero tool executions.

### S4. Only registered tools with schema-valid arguments may execute

- **Relevant production code:** immutable registry from `support_tools.registry.build_tool_registry()`; `agent.models.TOOL_SPECS` and strict argument models; `ToolExecutor.execute()` / `execute_named()`.
- **Enforcement layer:** Tool executor immediately before callable invocation.
- **Classification:** **Deterministic**.
- **Current failure behavior:** Unknown tool, non-object JSON, malformed JSON, missing/extra/wrong-type/blank/oversized arguments, registry/spec mismatch, or an unexpected result type yields a safe failed observation. Invalid calls do not invoke the handler.
- **Existing automated coverage:** Extensive coverage in `tests/test_agent_executor.py`, `tests/test_agent_models.py`, `tests/test_agent_runtime.py`, and `tests/test_support_tools.py::test_registry_is_immutable_and_contains_only_approved_tools`.
- **Obvious untested gap:** The executor is well covered in isolation; Step 2 should add an HTTP/graph case combining prompt injection with an allowlisted tool name but schema-invalid arguments, proving zero underlying invocation at the production boundary.

### S5. Tool observations and retrieved context are data, not control instructions

- **Relevant production code:** `build_rag_messages()` and `build_tool_messages()` JSON serialization and data notices; `RAG_SYSTEM_INSTRUCTIONS`; `BASE_SYSTEM_PROMPT` section 6; `build_agent_messages()`.
- **Enforcement layer:** Deterministic serialization/framing plus model instruction following.
- **Classification:** **Partial**, ultimately **LLM-dependent** for semantic behavior.
- **Current failure behavior:** Attacker text cannot syntactically escape JSON and become an actual system message, but the model can still choose to follow an instruction contained inside retrieved/tool data. No final semantic validator detects that outcome.
- **Existing automated coverage:** `tests/test_prompts.py::test_tool_observation_is_untrusted_data_with_route_policy`, RAG trust-rule tests, and `test_json_round_trip_keeps_untrusted_text_as_data`. Historical Day 6/7 evaluation includes sampled injected context, but that is evidence, not universal enforcement.
- **Obvious untested gap:** No production red-team suite injects instructions independently through retrieved chunks and successful tool observations across all relevant routes, including attacks asking for tool use, policy disclosure, and false commercial facts.

### S6. AI must not provide or confirm price, inventory, delivery time, warranty terms or contract terms

- **Relevant production code:** `prompts.BASE_SYSTEM_PROMPT` sections 2, 3, and 5; inventory/current-state patterns in `routing.router`; `SAFE_FALLBACK_ANSWER`; tool schemas and implementations, none of which expose these commercial fields.
- **Enforcement layer:** Router for a narrow inventory subset; capability design; primarily model policy for final wording.
- **Classification:** **Partial**, predominantly **LLM-dependent**. Known inventory/current-state phrasing can deterministically route to fallback, and tools do not return these fields, but price, delivery, warranty, contract terms, paraphrases, and mixed questions are not deterministically filtered from answers.
- **Current failure behavior:** Recognized current-inventory questions receive a fixed fallback without a model call. Other unsupported commercial questions rely on the router/model and prompt to abstain; a violating generated claim would not be removed by `render_answer()`.
- **Existing automated coverage:** Router tests cover current inventory fallback. Prompt-contract tests assert open-world commercial-fact rules. Evaluation data/reviews include sampled price, inventory, delivery, and unsupported-operation cases; these are not comprehensive deterministic production tests.
- **Obvious untested gap:** Step 2 needs direct and multi-turn cases for each named category—price, inventory, delivery time, warranty, and contract terms—plus mixed known-product facts and unsupported commercial facts, Chinese/English paraphrases, coercion, and false “previously confirmed” history.

### S7. False/nonexistent product premises must not be confirmed as facts

- **Relevant production code:** `ExactEntityResolver`; `HybridRouter.route()`; `DeterministicTools.get_product_details()`; `ToolExecutor`; epistemic rules in `BASE_SYSTEM_PROMPT` and `AGENT_TOOL_POLICY`.
- **Enforcement layer:** Exact identity resolution and authoritative tool lookup for exact-product paths; model policy on direct/RAG/agent final wording.
- **Classification:** **Partial**.
- **Current failure behavior:** Unknown or noncanonical product identifiers do not become exact catalog matches. `get_product_details` returns `PRODUCT_NOT_FOUND` or `AMBIGUOUS_PRODUCT`. The agent then receives a safe observation, but the final statement still depends on the model not inventing a product or converting absence of evidence into an overbroad claim.
- **Existing automated coverage:** Entity normalization/boundary/collision tests in `tests/test_entity_resolver.py`; unknown/display-name/ambiguous tool cases in `tests/test_support_tools.py`; `tests/test_chat_http.py::test_unknown_product_domain_error_is_observed_without_system_failure`; prompt epistemic-rule tests.
- **Obvious untested gap:** No systematic HTTP matrix tests fabricated but plausible models, near-miss IDs, product-name collisions, user-asserted specs, multiple real/false products, and false premises planted in conversation history.

### S8. Oversized or malformed input must fail safely at the appropriate boundary

- **Relevant production code:** `models.ChatMessage` / `ChatRequest`; `AgentToolCall` normalization; strict tool argument models; `Retriever.retrieve()` validation; FastAPI validation/error handlers.
- **Enforcement layer:** HTTP request validation, agent normalization, tool executor, and retriever.
- **Classification:** **Deterministic** for the defined limits and schemas.
- **Current failure behavior:** Invalid HTTP input returns safe 422 before orchestration. Invalid model completions terminate with a safe agent answer. Invalid tool calls produce `INVALID_ARGUMENT` without execution. Retrieval rejects blank queries or nonpositive `top_k` before embedding.
- **Existing automated coverage:** `tests/test_models.py` covers trimming, blank input, and role sequencing; `tests/test_chat_http.py` covers a whitespace-only 422; `tests/test_agent_models.py`, `tests/test_agent_executor.py`, `tests/test_agent_loop.py`, and `tests/test_retriever.py` cover malformed downstream inputs.
- **Obvious untested gap:** Explicit HTTP boundary tests are missing for 4,001-character messages, 21 messages, more than 20,000 total characters, missing/wrong-typed fields, invalid roles, malformed JSON/content type, and verification that each rejection causes zero orchestration/tool activity.

### S9. Rapid requests must be constrained by the existing rate-limit boundary

- **Relevant production code:** `rate_limit.enforce_rate_limit()`; dependency on `/api/chat-stream`; rate settings in `config.py`.
- **Enforcement layer:** FastAPI route dependency before orchestration.
- **Classification:** **Deterministic within one application process and one `request.client.host` identity**.
- **Current failure behavior:** The first ten requests in the 60-second window are recorded; further requests receive HTTP 429, `rate_limit`, a safe message, request ID, and `Retry-After`.
- **Existing automated coverage:** `tests/test_chat_http.py::test_validation_rate_and_concurrency_fail_before_orchestration` verifies a reduced one-request limit and the second request's 429 result.
- **Obvious untested gap:** No tests cover exact threshold/window expiry, `Retry-After`, independent client keys, missing client metadata, concurrent increments, or multi-process behavior. The implementation is process-local and its effective client identity depends on deployment/proxy handling; this is a documented boundary characteristic, not a redesign proposal.

### S10. Conversation history must not override the above policies

- **Relevant production code:** `ChatRequest.validate_role_sequence()`; prompt builders preserving history after system messages; `BASE_SYSTEM_PROMPT` section 4; `HybridRouter._historical_product_ids()` uses prior `user` messages only for deterministic product resolution.
- **Enforcement layer:** Role/ordering validation and model instruction hierarchy.
- **Classification:** **LLM-dependent**, with limited deterministic routing safeguards.
- **Current failure behavior:** Malformed role sequences are rejected. Validly shaped client-supplied history—including forged `assistant` content—is forwarded to the model. The application does not authenticate historical assistant messages or semantically inspect history for policy attacks.
- **Existing automated coverage:** `tests/test_models.py` covers role sequence; `tests/test_prompts.py` and `tests/test_agent_graph.py` verify history preservation; router tests cover contextual product references and repeated explicit current questions.
- **Obvious untested gap:** No adversarial history suite covers forged assistant approvals, prior fake commercial commitments, delayed instructions, cross-turn prompt extraction, history-vs-latest-query conflicts, or attacks split across multiple messages.

### S11. Failure responses must not expose stack traces, secrets, system prompts or internal configuration

- **Relevant production code:** `routes.chat._error_contract()` / `_http_exception()`; all handlers in `error_handling.py`; safe tool exception conversion in `ToolExecutor` and `DeterministicTools`; bounded request logging.
- **Enforcement layer:** Route error mapping, global exception handlers, tool boundary, and telemetry serialization.
- **Classification:** **Deterministic for application/provider/tool exception responses and current logs**. A model voluntarily disclosing hidden text is covered by the weaker S2 boundary instead.
- **Current failure behavior:** Known failures map to safe 502/503/504 contracts; unexpected exceptions map to safe 500 `internal_error`. Public output includes only code, generic message, and request ID. Tool observations use safe codes/messages. Logs use bounded trace fields and exception type rather than raw exception text or request/answer payloads.
- **Existing automated coverage:** `tests/test_error_handling.py`, failure cases in `tests/test_chat_http.py` and `tests/test_chat_route.py`, executor/service redaction tests, and log privacy tests.
- **Obvious untested gap:** Add a single production-boundary canary matrix that places recognizable secrets in router, embedding, graph, tool, rendering, and logging exceptions and asserts absence from response body/headers/events/log output for every mapped and unmapped path.

### S12. Security failures must not cause unauthorized tool side effects

- **Relevant production code:** fixed registry in `support_tools.registry`; read-only registered methods in `DeterministicTools`; validation-before-lookup/invocation in `ToolExecutor`; agent whole-batch rejection and maximum-call checks.
- **Enforcement layer:** Capability registry, executor, agent call-budget gate, and current tool implementations.
- **Classification:** **Deterministic for the current registered capability set**. The three registered tools only read local knowledge or perform retrieval; no mutating business capability is registered.
- **Current failure behavior:** Unknown/invalid calls never invoke a handler. Oversized or duplicate-ID batches terminate before tool execution. Tool failures are returned as safe observations. Valid repeated calls may be request-locally reused, and only actual calls count as executions.
- **Existing automated coverage:** `tests/test_agent_executor.py` proves unknown/invalid calls do not execute; `tests/test_agent_runtime.py` and `tests/test_agent_graph.py` cover duplicate IDs, atomic oversized-batch rejection, call budget, deadlines around tool execution, and request-local caching; registry tests prove only three approved tools are present.
- **Obvious untested gap:** Add end-to-end red-team cases asserting zero underlying invocations for injected destructive names, invalid registered-tool arguments, duplicate/oversized batches, and a deadline reached before execution. If a mutating tool is ever introduced, this invariant must be re-audited rather than inferred from today's read-only registry.

## 5. Deterministic versus model-dependent inventory

### Deterministically enforced now

- HTTP message structure, trimming, non-empty content, per-message/history/total size limits, and role ordering.
- Process-local rate and concurrency admission limits.
- Strict router-label parsing and fixed fallback on malformed classifier output.
- Exact product entity/catalog matching; user assertions do not add catalog records.
- Fixed registered tool set, strict JSON-object/Pydantic arguments, no extra fields, validated result types, and a maximum of three agent tool calls.
- No handler invocation for unknown or invalid tool calls.
- Current tool surface is read-only and based on authoritative local inventory/retrieval.
- Safe structured tool errors and safe public HTTP error contracts.
- Raw URL/reference-section removal, trusted citation-source validation, and safe fallback when sanitization empties an answer.
- Bounded telemetry that excludes request text, answer text, retrieved text, tool arguments/observations, and raw exception messages.

### Still dependent on model behavior in whole or in part

- resisting semantic instruction override from user text;
- refusing to reveal system/hidden prompts through generated prose;
- treating retrieved context, tool observations, and validly shaped history as data rather than instructions;
- refusing unsupported price, delivery, warranty, contract, and paraphrased inventory claims;
- not confirming false premises outside exact deterministic lookup outcomes;
- preventing validly shaped forged history from influencing policy compliance;
- avoiding other unsupported factual claims in final answer text.

`render_answer()` is not a general policy firewall. It enforces URL/reference and trusted-citation boundaries, but it does not inspect commercial claims, prompt fragments, factual grounding, or instruction-following semantics.

## 6. Current test coverage summary

Current automated tests are strongest around deterministic mechanics:

- request and role validation: `tests/test_models.py`, `tests/test_chat_http.py`;
- rate/concurrency admission and slot lifecycle: `tests/test_chat_http.py`, `tests/test_chat_route.py`;
- router behavior and strict classifier output: `tests/test_router.py`;
- exact entity resolution and retrieval validation: `tests/test_entity_resolver.py`, `tests/test_retriever.py`;
- prompt construction/data envelopes: `tests/test_prompts.py`;
- tool allowlist, schemas, error redaction, caching, and non-execution: `tests/test_agent_models.py`, `tests/test_agent_executor.py`, `tests/test_agent_runtime.py`, `tests/test_support_tools.py`;
- LangGraph execution, call budgets, deadlines, and fallbacks: `tests/test_agent_graph.py`, `tests/test_graph_route_orchestrator.py`;
- answer/source sanitization: `tests/test_citations.py`, `tests/test_incremental_sanitization.py`, `tests/test_chat_route.py`;
- public error and logging redaction: `tests/test_error_handling.py`, `tests/test_chat_http.py`;
- sampled semantic smoke/evaluation: `tests/test_day7_production_smoke.py`, `tests/test_day7_trust_boundary.py`, and retained `eval/results/` artifacts.

Prompt-contract tests establish that the intended rules reach the provider request. They do not establish that every model response obeys those rules. Likewise, historical smoke artifacts demonstrate sampled outcomes, not universal enforcement.

## 7. Concrete Step 2 red-team case backlog

The following gaps should become focused red-team cases without redesigning production:

1. **Direct prompt override/exfiltration:** identity override, fake administrator authority, policy hierarchy extraction, system prompt/config/API-key requests, encoding and multilingual variants.
2. **History attacks:** forged assistant approval, prior fake price/warranty/contract commitment, delayed instructions, attacks split across turns, and latest-message conflict with history.
3. **Retrieved-context attacks:** injected instructions to change facts, reveal prompts, call tools, redirect contacts, or assert commercial terms; verify retrieved text stays data.
4. **Tool-observation attacks:** malicious-looking strings inside successful and failed observations; verify they cannot select additional tools or alter policy.
5. **Commercial-policy matrix:** price, inventory, delivery time, warranty, and contract terms, each as direct, coercive, hypothetical, translated, mixed-known/unknown, and history-based questions.
6. **False-premise matrix:** nonexistent plausible model, near-miss ID, user-supplied fake specification, one real plus one fake model, and false facts asserted by prior assistant history.
7. **Tool authorization:** fake function-call syntax in user text, unknown destructive tool, allowlisted name with invalid JSON/schema/extra fields, duplicate IDs, and over-budget batch; assert zero unauthorized underlying invocations.
8. **Input-boundary matrix:** exact and over-limit sizes, message count and total length, malformed JSON, wrong types/roles/content type; assert safe 422 and zero orchestration/tool execution.
9. **Rate-limit behavior:** exact threshold, window rollover, `Retry-After`, client-key isolation, missing client metadata, and concurrent requests within one process.
10. **Failure canaries:** recognizable secret/traceback/prompt fragments injected into router, embedding, tool, graph, rendering, and logging failures; assert absence from all public and logged surfaces.

These cases should separately report deterministic boundary failures and semantic/model failures so that a model miss is not misclassified as an allowlist or validation defect.

## 8. Files inspected

### Production request and policy path

- `main.py`
- `models.py`
- `config.py`
- `rate_limit.py`
- `concurrency.py`
- `routes/chat.py`
- `error_handling.py`
- `prompts.py`
- `trace_models.py`
- `routing/router.py`
- `routing/deterministic.py`
- `routing/models.py`
- `routing/generation.py`
- `routing/knowledge.py`
- `routing/finalization.py`
- `agent/models.py`
- `agent/runtime.py`
- `agent/executor.py`
- `agent/loop.py`
- `agent_graph/graph.py`
- `agent_graph/nodes.py`
- `agent_graph/orchestrator.py`
- `support_tools/registry.py`
- `support_tools/models.py`
- `support_tools/service.py`
- `services/llm.py`
- `services/retrieval.py`
- `services/tools.py`
- `knowledge_pipeline/retrieval/entities.py`
- `knowledge_pipeline/retrieval/retriever.py`
- `knowledge_pipeline/retrieval/models.py`
- `citation/core.py`
- `citation/sanitization.py`

### Existing test/evidence files

- `tests/test_models.py`
- `tests/test_chat_http.py`
- `tests/test_chat_route.py`
- `tests/test_router.py`
- `tests/test_deterministic_routing.py`
- `tests/test_entity_resolver.py`
- `tests/test_retriever.py`
- `tests/test_prompts.py`
- `tests/test_agent_models.py`
- `tests/test_agent_executor.py`
- `tests/test_agent_runtime.py`
- `tests/test_agent_loop.py`
- `tests/test_agent_graph.py`
- `tests/test_graph_route_orchestrator.py`
- `tests/test_support_tools.py`
- `tests/test_citations.py`
- `tests/test_incremental_sanitization.py`
- `tests/test_error_handling.py`
- `tests/test_day7_production_smoke.py`
- `tests/test_day7_trust_boundary.py`
- `eval/user_prompt_injection_v1.json`
- `eval/results/day7-week3-production-smoke-v1.jsonl`
- `eval/results/day7-step4-regression-review.json`

## 9. Scope confirmation

This Step 1 audit adds only this documentation file. It does not modify production code, middleware, dependencies, evaluation behavior, or tests.
