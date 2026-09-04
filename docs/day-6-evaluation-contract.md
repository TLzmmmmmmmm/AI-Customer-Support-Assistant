# Day 6 Step 1：评测契约与 ground truth

状态：Step 1–4已完成；Step 5报告与评分已由用户确认，Step 5 pass、review_status=complete。Step 6失败分类见`day-6-step-6-failure-analysis.md`，只记录、不修复。Step 7已运行20道frozen并完成同题比较报告，V1 frozen评分经人工核对已complete，V0独立复评分仍pending_review。Step 8已在配置冻结后运行10道holdout一次，评分经人工核对已complete；最终报告见`rag-evaluation-v1.md`。已知缺陷未修复，V1评分审核已完成，V0独立复评分仍待确认，整体产品质量验收未通过。本文保留早期步骤历史记录，当前范围以以上最新执行状态为准。

## 最新执行范围（Step 6开始时用户确认）

- Day 6只负责评测、发现、分析和记录问题；全部产品行为、prompt、检索及数据修复统一留到Day 7。
- 用户批准Step 5通过并确认既有报告/评分。阶段通过不重写失败回答、不提高分数、不表示缺陷已修复。
- 已确认评分文件review_status为complete；原始运行JSONL的pending_review保留为运行时状态，后续审核状态以独立评分文件为准。
- Step 6继续按源资料→标准化文档→chunk→检索→上下文→生成→策略的顺序定位；修复建议只形成Day 7待办，不在Day 6实施。
- 后续Step 7/8仍分别等待明确指示；Day 6不调优，holdout应评测实际冻结的当前配置。早期文档中“开发调优后”的建议不构成本日修改授权。

## 1. 当前生效的验收口径

- 用户已暂定 Day 5 通过。这是验收决定，不是新增测试结果，也不把旧的 22/32 直接改写成新的准确率。
- 自动咨询建议和重复总结可以接受；它们不能增加错误事实、未经支持的因果解释、价格、承诺或能力声明。简洁是偏好，不再单独以重复或咨询建议判定失败。
- 英文问题须全英文回答；若问题明确指定另一回答语言，尊重该显式要求。公司英文品牌和全称须准确，普通英文文风不苛求润色。
- Groundedness 认可实际交给生成模型的检索证据，以及应用明确提供、经公司负责人确认的系统身份信息和业务规则。二者分开记录，不能把未检索到的 KB 内容冒充本次生成依据。
- 型号、具体参数、认证代码、商业信息仍须有对应依据。未知事项要明确无法确认，不以“可能”“通常”或免责声明掩盖猜测。
- 当前 production prompt 仍比新验收口径严格地禁止主动咨询/总结；Step 1 不调 prompt。保留配置测量，是否进一步放宽在开发评测后单独批准。

## 2. 现有文件与案例数量

Step 1 开始时，docs 与 eval 共 70 个文件：24 个 Markdown、6 个 JSON、40 个 JSONL。

| 材料 | 内容 / 状态 | Day 6 用途 |
| --- | --- | --- |
| eval/baseline_v0.json | 20 条冻结题，包含 20 份已完成的 V0 原始回答、逐题分数；17 可回答、1 部分可回答、2 不可回答 | 永久保留原件，V0/V1 同题比较 |
| eval/baseline_v0_results.json | V0 汇总与运行元数据 | 永久保留原件，不重跑覆盖 |
| eval/retrieval_v1.json | Day 4 原版 18 题，含 all_groups 标注 | 历史标注参考，不是 18 条新独立问题 |
| eval/retrieval_v1_1.json | 更新后的 18 题，映射 baseline-001..017 与 baseline-019 | chunk ground truth 起点，逐条人工再核验 |
| eval/retrieval_v1_results.json / retrieval_v1_1_results.json | 两次旧检索汇总 | 已批准移入回收站；不作为 Day 6 分数 |
| docs 中原有 40 个 JSONL | 309 条 case 记录、41 个完全不同的 query 字符串；含 1 次传输失败，其余为重复开发运行 | 已批准移入回收站；不能按 309 条或 41 条直接计入新冻结数据集 |
| scripts/day5_abstention_eval.py | 可复用的开发问题与场景定义 | 后续筛选为 dev，不能重新称为未见 holdout |
| docs 中架构/契约/设计/实施说明 | 现有接口、业务规则和失败线索 | 保留供理解项目，不作为当前评测成绩 |

V0 原始文件在当前 checkout 中未被 Git 跟踪且被 eval/ 忽略，不能假设 Git 能恢复它们。不得删除、移动、覆盖或顺手格式化。

## 3. 最小案例 schema 提案

不修改旧 Day 4 Pydantic 类型；新 Day 6 契约单独版本化。Step 1 只定义结构，不实现模型、runner、指标或完整数据集。

| 字段 | 类型 / 必填 | 用途与约束 |
| --- | --- | --- |
| id | string，必填 | 全数据集唯一；原 20 题保留 baseline-001 等 ID，连接历史 V0 输出 |
| category | enum，必填 | product_spec（含对比）、product_recommendation、solution、support、company（含联系）、unknown、adversarial |
| split | enum，必填 | frozen / dev / holdout；角色不等同于是否可回答 |
| question | string，必填 | 实际用户问题；冻结基线题的原文保持不变 |
| expected_behavior | enum，必填 | answer / partial_answer / abstain / clarify；避免另设重复的 answerability 字段 |
| expected_answer | string，必填 | 人可读的正确回答要点或适当拒答/澄清方式，不做逐字匹配 |
| expected_document_ids | string[]，必填，可空 | 源 document 的标识；仅用于溯源，不替代 chunk 命中 |
| expected_chunk_ids | string[]，必填，可空 | 回答所需 chunk 的 ground truth，不是整篇文档内所有 chunk |
| expected_system_rule_ids | string[]，必填，可空 | 使用的可信系统事实/业务规则，独立于 chunk IDs |
| required_facts | string[]，可选，默认 [] | 必须包含的事实或拒答边界，供人工语义复核和有限辅助检查 |
| forbidden_claims | string[]，可选，默认 [] | 不得出现的错误或无依据结论，不是禁词表 |
| fixture_id | string，可选 | 仅受控的检索内容注入等场景使用；指向独立的测试 fixture，不污染生产 KB |

id 与问题字符串不同：同一问题可能在不同版本重跑，但同一 dataset 不能复制成多个 ID 虚增样本数。分类选择主测试目的，不把生产 RetrievalResult.type 当作评测类别。

运行信息不放进案例：dataset/schema 版本、知识快照、系统规则版本在数据集级别记录；模型、Top-K、prompt hash、commit、dirty 状态/相关文件 hash、时间和每次实际 hits/回答留给 Step 3/5/7 的 run metadata。Git commit 不能单独代表当前未提交的 prompt。

## 4. 可回答、未知与纯系统题

| 情形 | expected_behavior | chunk / system 依据 | 检索计分 |
| --- | --- | --- | --- |
| KB 有全部所需信息 | answer | 非空 expected_chunk_ids；系统规则可辅助 | 参与 |
| 部分属性已知、部分未知 | partial_answer | 标注已知部分所需 chunks；未知部分写入预期与禁止猜测项 | 有 chunk ground truth 时参与 |
| 所问信息没有可信依据 | abstain | 可为空；expected_answer 明确缺失信息与不得猜测 | 无 expected chunks 时不计入 Hit/Recall |
| 只问官方英文名 | answer | expected_chunk_ids=[]；expected_system_rule_ids=[company_identity_v1] | 不参与；不能把空集打成 0 或 1 |
| 型号分类+具体参数 | answer | 型号身份/参数 chunks + radio_classification_v1 | 只对 chunk 部分计检索，系统规则单独核验 |
| 对象不明确、需澄清 | clarify | 按案例需要，可为空 | 无 chunk ground truth 时不参与 |

可回答不等于必须有 expected chunks；没有 expected chunks 也不等于不可回答。不能用检索低分、没命中或“页面没写”单独证明知识缺失。未知题须先核对当前公司资料及系统规则。

Ground truth 应选直接含所需证据的最小集合。多事实题可能需要多个 chunk；命中文档中的无关章节不算命中。若存在真实等价证据，应在冻结标注前解决，不能在看到失败后随意增加标签“修好”成绩。旧 all_groups 的分组分数不得直接冒充 Day 6 平铺 chunk Recall；若无法用明确 chunk 集合合理表达，记录标注歧义并改进题目，而非偷偷改变指标定义。

fixture_id 场景中，干净支持事实的预期 chunk 与人为注入内容分开。未来 runner 必须标记 synthetic context；这些结果可以测试信任边界，但不能当成未经干预的生产端到端检索结果。Step 1 不创建 fixture。

## 5. 当前系统事实与规则的版本化登记

以下是评测依据的登记，不是新增生产功能；未来 dataset 引用这些稳定 ID，并随实际规则变更递增版本。

### company_identity_v1

- Brand: Shengborun Communications
- Full English Name: Beijing Shengborun Communication Equipment Co., Ltd.
- 依据：公司负责人明确确认；现行 prompts.py 身份部分。

### radio_classification_v1

- 适用范围：公司已确认的全部对讲机产品，不外推摄像机或配件。
- 正式名称以 Ex、CQST 或“防爆”结尾的为防爆对讲机；其余为非防爆。后缀前空格不影响。
- 使用正式型号，不因用户编造或添加后缀而认定公司有该产品。
- 仅给出分类；具体防爆等级、认证编号和设计原因仍需产品证据。
- 依据：公司负责人明确确认；现行 prompts.py 业务规则。

验收策略（consultation/summary 可接受、英文要求、拒绝无依据推测）是 rubric 版本，不是可用于编造产品答案的事实来源。V0 原运行不曾收到的新系统规则不能被倒填成 V0 的实际输入。

### 当前不添加官网链接或联网功能（Step 4开始前用户修订）

用户撤销Step 3新增的官网链接及`company_website_v1`规则；已从system prompt和manifest移除。现阶段明确不添加网页访问功能。既有知识来源与provenance保持不变，不修改题集、历史V0或holdout标签。

## 6. V0 原件与 Day 6 标注分离

原件及哈希保持不变，禁止把新预期或重新评分写回原文件。

已确认冲突：baseline-004 旧标注明确禁止“HP780 一定不防爆”；现行业务规则则确认 HP780 非防爆。Step 2 保留同一问题和 ID，在新的 Day 6 数据集中单独写入现行 ground truth，并登记该标注变更及理由。

Step 7 的比较分为：

1. 原始 V0 历史成绩：原样引用并标明旧规则/旧资料；
2. 同题、同一 Day 6 rubric 下的新复评：读取旧 V0 回答，不重新生成，再与新 V1 回答比较；
3. 每个系统的 groundedness 按实际可用的证据分别核对。无法恢复 V0 当时完整输入的项目标为不可核验，而不是假设其见过今天的全部 KB/系统规则。

Correctness 可依据最新权威知识统一复评；Groundedness 衡量“答案是否由该次可用依据支持”，两者不能混同。报告知识、规则与 prompt 的变化，不能将全部差异归因于 RAG。没有历史 V0 回答的新 dev/holdout 题不伪造 V0 对照分数。

## 7. 六维出题审查（Step 2 执行）

使用用户提供的三档人工审查，不添加六套自动评分，也不要求每题在每一维都机械拿最高档。

| 维度 | 三档 | 审查重点 |
| --- | --- | --- |
| Naturalness | 很人工 / 尚可 / 很像真实用户 | 自然客户表达，不照抄 chunk |
| Ground truth | 模糊 / 部分明确 / 非常明确 | 应答边界、必需事实、禁止猜测明确 |
| Retrieval value | 纯关键词 / 有少量改写 / 真正语义改写 | 保留精确实体测试，同时确保语义改写覆盖 |
| Evidence alignment | 不支持 / 部分支持 / 明确支持 | 可回答部分确有证据；未知题明确证实当前可信知识不足 |
| Diagnostic value | 很难归因 / 尚可 / 明确测某能力 | 清楚定位知识/检索/上下文/生成/策略问题 |
| Ambiguity | 很歧义 / 有一点 / 几乎无歧义 | 普通题尽量消歧；有意测试澄清的题明确注明 |

在独立 authoring review 中记录六维判断；不膨胀最小案例 schema。旧 Day 5 已使用的问题只能进入 dev，不能重新命名为未见 holdout。用户已批准共 60 题：20 条原始冻结基线、30 条开发题、10 条全新 holdout。Step 1 的数量盘点不是数据集构建。

## 8. 现有 ID、runner 与测试兼容性

- 当前 61 个 document_id、73 个 chunk_id 均唯一；没有孤立 parent_document_id。
- 两版 Day 4 的所有 expected chunk/document IDs 当前仍存在。但原版 retrieval_v1.json 的整库 hash 已过时；v1.1 的 hash 与当前 chunks 匹配。ID 存在不保证证据充分。
- Day 6 expected_document_ids 对应 production RetrievalResult.parent_document_id 和 KnowledgeDocument.document_id；不要改生产字段名。
- Day 6 question 对应旧 Day 4 query；split/expected_behavior/系统依据是新评测层字段，不塞进 Retriever。
- 旧 RetrievalEvaluationCase 强制非空 expected_chunk_ids，不能直接覆盖新契约；旧指标包含 MRR、entity、多源分组指标，不能把它的汇总直接当成 Day 6 三个简单指标。
- production 对象实际名为 RetrievalResult，概念上就是文档中的 RetrievalHit。rank、score、match_origin、chunk_id、parent_document_id、text、content_hash、metadata、source_url、source_files 都已在后端可用。
- Context Builder 只把 type/section/text 传给生成；后端完整 provenance 仍保留。模型实际见到的 context 需在受控 eval 中追踪，但不得改变 production logging。
- tests 使用 unittest；现有脚本默认预检，--execute 才调用付费 API。Step 1 不调用任何 provider。
- eval/ 与 docs/ 在 .gitignore 中；新契约/标注/结果后续需要显式纳入版本管理或约定存储位置，不能误认为普通 git add 已保存。

## 9. 快照与清理边界

SHA-256（Step 1 盘点时）：

- eval/baseline_v0.json: c20b295a404c9aa9996075e844002be6a9a5d0298b3dc58d1e5bf19cef583160
- eval/baseline_v0_results.json: 01d9b0e255446a50cee5cc717ad32ad4ec21de662b154ae8ea8d6615f972275d
- knowledge/documents.jsonl: adb239c90ce61e241e791936e79ef46b404afc1fa0582e6928ba90dca900e2c3
- knowledge/chunks.jsonl: e39d3869d4a4be6f8f4303cc15ab2d3d5d22473442e977dc40a9bc2481c1c490

具体清理名单见 [Step 1 清理清单](day-6-step-1-cleanup.md)。用户明确批准后，42 个旧结果已移入回收站，并核对全部 42 条回收站记录；V0 原件未变。旧结果删除不等于 Step 6 失败分析已完成。Day 6 将重新运行、定位并记录当前结果。

## 10. 本步完成边界

本次新增契约和清理清单，更新 Day 5 的当前验收状态及五份历史实验文档的清理说明，并按批准移除 42 个旧结果。只调整一个测试的旧结果文件存在要求，保留历史题集检查和全部检索功能测试。没有新增/重跑正式评测题，没有实现 schema 验证器或指标，没有修改生产 prompt、Retriever、API、日志或前端。清理后报告并停止，不进入 Step 2。

已执行的验证：

- 清理前现有全套 unittest：183 tests，OK；没有新增测试方法。
- 清理前 6 个 eval JSON 和 40 个旧 JSONL 均可解析；基线 20 个 ID 唯一且均有已完成的 V0 回答。
- 61 个 document / 73 个 chunk 的 ID 唯一，parent 引用无孤立项；两版旧题集的全部 ID 均存在。未声称已完成 Step 2 的逐条证据充分性审核。
- 清理清单：事前验证 42 个唯一文件全部存在，无 V0 保护文件，共 2,465,362 字节；执行后全部不在原路径，且 42 个唯一文件在回收站中核实存在。
- V0 两文件哈希与本步开始一致；git diff --check 通过。
- 批准清理后复测：检索评测模块 10 tests，OK；全套 183 tests，OK。先观察到旧存档测试因旧结果不存在而失败，再按批准调整其存档要求；没有隐藏检索行为失败。
- 清理后 V0 的 20 条问题与 20 份已完成原始回答仍在；V0 两文件、documents、chunks、vector_records 五个文件的哈希均与操作前一致。
- 清理后 docs/ 与 eval/ 共保留 30 个文件，42 个目标全不在原路径、全在回收站核实；V0 与两版旧题集均保留。五份历史实验说明已标注清理，不把旧分数充当 Day 6 成绩。

复现本地测试（沿用用户允许的 .venv 依赖）：

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv/Lib/site-packages').Path
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest discover -s tests
```

该测试不运行正式付费评测。新契约和清单位于被忽略的 docs/ 中，本次未提交 Git。

## 11. Step 2 开始前已批准的补充决策

- 题量：20 frozen + 30 dev + 10 holdout。历史基线原始文件不改；当前标注单独存储。holdout 可以在出题时检查依据，但不得在 Step 8 之前运行或根据表现调参。
- 评分责任：助手逐题组织证据并提出评分建议，用户确认失败、争议案例及评分汇总。未确认的分数标记待审核，不称为已完成人工审核；不增加 LLM judge。
- Correctness / Groundedness / Completeness / Refusal Quality 使用 0、1、2，分别为不符合、部分符合、完全符合。Hallucination 单独记有/无并指出问题，不与上述分数混合方向。
- 不适用或无法评估记 N/A，必须说明原因，并从相应指标分母排除。例如完全可回答的问题无需评拒答质量，无法还原输入依据的 V0 回答不强行评 Groundedness。
- 各指标单独报告，不合并为总分。当前 Step 2 不生成回答、不评分；出题六维审查也不是模型质量分数。
- 评估工程完成与模型质量通过分开。题集、流程、对比和报告完成，不代表模型无问题；发现问题也不代表评估工程失败。
- 严重产品参数错误、防爆分类错误、编造认证或因果关系、泄露内部信息或服从恶意指令，阻断质量通过。
- 遗漏、过度拒答等一般问题逐项报告给用户审核；事实正确的咨询建议及重复总结允许。
- 报告实际分数和样本数，不设置缺乏依据的统一90%通过线。验收规则在holdout运行前固定，不根据结果放宽。
- Step 2 为新题集及三份 Day 6 文档添加定向 Git 忽略例外，不提交文件、不改变历史 V0 原件或旧结果的忽略策略。
