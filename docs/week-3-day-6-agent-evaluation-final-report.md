# Week 3 Day 6 — Agent Evaluation Final Report

## 1. Objective

本次评测用于验证当前 LangGraph Agent 架构在以下三个层面的正确性：

1. **Routing**：是否将用户请求路由到正确的执行路径。
2. **Action**：是否调用正确的工具，并满足工具调用数量和参数要求。
3. **End-to-End Answer Quality**：最终回答是否符合冻结的事实、能力边界与回答要求。

本次评测不覆盖：

- Citation Coverage
- Latency
- Token Usage
- Streaming Performance
- Retrieval Ranking Metrics
- Failure Taxonomy
- LLM-as-a-Judge

评测采用：

```text
Dev24
→ 开发阶段发现并修复问题
→ 冻结最终 Dev 结果

Holdout6
→ 未见测试
→ 仅执行一次
→ 不根据 Holdout 结果继续调优后重新计算 Holdout
```

## 2. Evaluation Architecture

当前生产路径包含六种 Route：

```text
direct
product_search
exact_product
contact
knowledge
fallback
```

评测层通过 `AgentEvaluationRunner` 观察：

```text
RouteExecutionResult
├── trace.route
├── validated tool calls
└── final_answer
```

自动计算两个 deterministic metrics：

```text
Route Accuracy
Action Accuracy
```

最终回答由人工按照冻结 rubric 评分：

```text
2 = Fully Correct
1 = Partially Correct
0 = Incorrect
```

### 2.1 Evidence Artifacts

本报告基于以下冻结数据集与正式执行结果：

```text
Dev dataset
evaluation/agent_dev_v1.jsonl
SHA-256: c5f9d7db80135cb31d98417069405519269d01b6d7c0ca1db5fe000887b4e25f

Dev result
eval/results/agent-dev-v1-20260912T132512Z-1062bf8f.json
SHA-256: 5a6d97a75c635aa1d894920410ea8eebbbffb2b686020b5e78d74600a414c71f

Holdout dataset
evaluation/agent_holdout_v1.jsonl
SHA-256: 65076bbb70b339a73b36148e8359dc1fb30320c6ae032d5d9f9b348255eac1ce

Holdout result
eval/results/agent-holdout-v1-20260912T134837Z-e56c7b01.json
SHA-256: 2e156bcda7cacb1cce9df40852b4ddae418a13fb50bc65603547212658609519
```

## 3. Dev24 Dataset

Dev V1 共包含：

```text
24 cases

direct          4
product_search  4
exact_product   4
contact         4
knowledge       4
fallback        4
```

Dev dataset 在正式执行前冻结。

正式运行后，不通过修改 golden expectation 来提高分数。

## 4. Dev24 Results

### 4.1 Route

```text
Route Accuracy
24 / 24
100.00%
```

六种 route 均成功通过最终 Dev 测试。

### 4.2 Action

Evaluator 的原始大小写敏感比较报告：

```text
Raw Matcher Result
23 / 24
95.83%
```

唯一原始比较失败：

```text
dev_exact_04
```

Golden expectation：

```text
get_product_details(product_id="ly198")
```

实际调用：

```text
get_product_details(product_id="LY198")
```

生产 Tool 层能够正确解析该 ID，并成功取得 LY198 数据。因此，`ly198` 和 `LY198` 在生产业务语义上表示同一个 canonical product。

该 case 的：

```text
Route         PASS
Tool Name     PASS
Multiplicity PASS
Entity        PASS
Execution     PASS
Final Answer  PASS
```

原始比较失败来自 evaluator 对字符串参数进行大小写敏感比较，而不是 Agent 行为错误。经人工 adjudication 后，正式记录为：

```text
Action Accuracy
24 / 24
100.00%
```

## 5. Dev24 End-to-End Evaluation

最终人工评分：

```text
E2E Score
47 / 48

Average Score
1.958 / 2.000
97.92%

Perfect Answers
23 / 24
95.83%
```

唯一非满分：

```text
dev_product_search_04
E2E = 1 / 2
```

该回答中的产品型号与参数均正确，但加入了：

```text
“酒店内部日常通信通常不需要防爆机型”
```

这一轻微的 scene-selection inference。该信息不是当前 source/tool evidence 明确支持的事实。

人工审核认为该问题 non-blocking，因此不继续针对该 Dev case 修改 production。

## 6. Important Dev Fixes

Dev evaluation 实际发现了两个有价值的问题。

### 6.1 Contextual Exact Product Resolution

原始失败：

```text
User:
LY198 的功率是多少？

Assistant:
≤2W

User:
那它支持什么频段？
```

原系统：

```text
“它”
→ ExactEntityResolver 无显式型号
→ fallback
```

修复后：

```text
contextual reference
→ history-aware routing / agentic resolution
→ exact_product
→ get_product_details("LY198")
→ 400-480MHz
```

最终成功验证：

```text
explicit entity
→ deterministic resolution

implicit conversational entity
→ history-aware / agentic resolution
```

### 6.2 Exhaustive Knowledge Enumeration

原始：

```text
你们有哪些解决方案？
```

普通 Top-K RAG 只返回部分 solution，因此最终只列出两个方案。

修复后能够完整返回当前六个解决方案：

```text
酒店行业无线对讲解决方案
企事业单位行业无线对讲解决方案
石油石化行业无线对讲解决方案
人防行业宽带自组网解决方案
宽带自组网应急管理行业解决方案
智慧应急解决方案
```

最终 Dev case 通过。

## 7. Holdout6 Design

Holdout V1 在 Dev 修复完成后建立，共六条：

```text
direct          1
product_search  1
exact_product   1
contact         1
knowledge       1
fallback        1
```

Holdout 不简单改写 Dev prompt，而是增加新的能力维度：

| Route | Holdout 新覆盖维度 |
| --- | --- |
| direct | Conversational acknowledgement |
| product_search | 防爆约束 + 完整 raw query |
| exact_product | 双实体 + 两次 tool call |
| contact | English end-to-end |
| knowledge | 单一 solution 内部精确事实 |
| fallback | 未实现的商业执行操作 |

Holdout 只进行了一次正式 provider execution。

## 8. Holdout6 Results

### 8.1 Route

```text
Route Accuracy
6 / 6
100.00%
```

全部六种 route 正确。

### 8.2 Action

Evaluator 的原始大小写敏感比较报告：

```text
Raw Matcher Result
5 / 6
83.33%
```

唯一原始比较失败：

```text
holdout_exact_01
```

Golden：

```text
get_product_details("ly598")
get_product_details("ly198")
```

实际：

```text
get_product_details("LY598")
get_product_details("LY198")
```

Agent 正确完成：

```text
2 entities
→ 2 tool calls
→ 2 observations
→ one aggregated answer
```

最终答案正确：

```text
LY598
Output Power: ≤5W
Battery: 2200mAh

LY198
Output Power: ≤2W
Battery: 1200mAh
```

该原始比较失败与 `dev_exact_04` 相同，属于 evaluator 的 case-sensitive product ID false negative。经人工 adjudication 后，正式记录为：

```text
Action Accuracy
6 / 6
100.00%
```

## 9. Holdout6 End-to-End Evaluation

所有六条最终回答均符合 frozen rubric：

```text
Manual E2E Score
12 / 12
100.00%

Perfect Answer Rate
6 / 6
100.00%
```

各 Route：

```text
DIRECT          PASS
PRODUCT_SEARCH  PASS
EXACT_PRODUCT   PASS
CONTACT         PASS
KNOWLEDGE       PASS
FALLBACK        PASS
```

其中尤其验证了：

```text
multi-tool exact-product execution
English contact response
constrained product discovery
solution-level knowledge retrieval
unsupported-action fallback
```

## 10. Final Metrics

### Dev24

```text
Route Accuracy
24 / 24 = 100.00%

Action Accuracy (adjudicated)
24 / 24 = 100.00%

Manual E2E
47 / 48 = 97.92%

Perfect Answer Rate
23 / 24 = 95.83%
```

### Holdout6

```text
Route Accuracy
6 / 6 = 100.00%

Action Accuracy (adjudicated)
6 / 6 = 100.00%

Manual E2E
12 / 12 = 100.00%

Perfect Answer Rate
6 / 6 = 100.00%
```

## 11. Descriptive Combined Results

Dev 与 Holdout 的角色不同，因此不能把 Combined metric 当成独立泛化指标。

仅作为描述性统计：

```text
Total Cases
30

Route
30 / 30 = 100.00%

Action (adjudicated)
30 / 30 = 100.00%

Manual E2E
59 / 60 = 98.33%

Perfect Answers
29 / 30 = 96.67%
```

原始 matcher 的两个 false negatives：

```text
dev_exact_04
holdout_exact_01
```

均属于同一个 evaluator canonicalization 问题，而非生产 Agent action failure。

## 12. Known Non-Blocking Issues

### 12.1 Evaluation Product-ID Canonicalization

当前：

```text
ly198 != LY198
```

会被 evaluator 的原始 matcher 判为参数不匹配，但 production 将两者解析为同一个 product entity。

后续可以让 evaluation observation 或 matcher 使用 production-compatible canonicalization。该修改属于 evaluation correctness improvement，而不是 Agent production fix。

Holdout 不应因此重新执行并重新计算为新的 Holdout result。

### 12.2 Minor Scene-Selection Inference

`dev_product_search_04` 中存在轻微 unsupported inference：

```text
“酒店内部日常通信通常不需要防爆机型”
```

人工判断该问题 acceptable、non-blocking。当前不为追求 Dev 100% 而进一步修改 production。

## 13. Holdout Status

Holdout V1 已经正式使用：

```text
agent_holdout_v1
status = consumed
```

后续即使修复 evaluator，或者修改 Agent、Router、Prompt、Retrieval，也不能再次运行相同六条并把结果称为新的 unseen Holdout performance。

如果需要下一轮真正的泛化测试，应建立：

```text
Holdout V2
```

并使用新的未见 cases。

## 14. Final Assessment

当前 Week 3 Agent 架构通过本轮 Minimal Agent Evaluation。

最终证据：

```text
Dev24:
100% Route
100% Action
97.92% E2E

Holdout6:
100% Route
100% Action
100% E2E
```

本轮评测验证了：

- 六类 Route 能够稳定区分；
- deterministic 与 agentic execution 能够协同；
- explicit 与 contextual product references 能够正确处理；
- product discovery 能遵守工具和事实边界；
- exact-product 路径能够执行多个工具调用并聚合结果；
- knowledge 路径能够处理 overview 与 solution-level factual queries；
- contact 路径能够跨语言工作；
- unsupported actions 能够安全进入 fallback；
- LangGraph production path 在当前测试范围内没有发现 blocking regression。

因此：

```text
Week 3 Day 6
Minimal Agent Evaluation & Regression

STATUS: PASS
```

当前不存在阻止进入下一阶段的已知问题。
