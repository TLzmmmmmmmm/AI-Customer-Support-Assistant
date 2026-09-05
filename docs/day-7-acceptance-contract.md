# Week 2 Day 7：RAG V1.1 可靠性加固验收合同

状态：Step 1 基线封存完成；尚未开始 Step 2 实验或任何生产行为修改。观察时间：2026-09-05 11:37:50（Asia/Shanghai）。

## 1. 权威起点与边界

Day 6 交付提交为 `99ac20959715d02ebcb7f112791cbc1c1e29d6bb`（分支 `week2-RAG`，提交说明 `Complete Day 6 evaluation and confirm V1 reviews`）。Day 6 数据集版本为 `rag-v1.0`，冻结规模为 61 个 normalized documents、73 个 chunks、73 个 vector records。

当前生产路径保持为 `POST /api/chat-stream` → Retriever → Context Builder → DeepSeek streaming，响应仍为 `application/x-ndjson` 的 `delta` / `done` / `error` 事件。Retriever 仍仅使用最后一条用户消息；配置仍为 DeepSeek `deepseek-v4-flash`、DashScope `qwen3.7-text-embedding` 1024 维、NumPy exact cosine + exact entity、Top-K=5。本步没有增加联网、改写 query、reranking、BM25/hybrid、vector database、阈值、citation 或生产 retrieval telemetry。

Day 6 的 dev、frozen、holdout V1 review 均经负责人人工核对并标记 `complete`。V0 独立 Day 6 复评分仍为 `pending_review`，不影响本合同所列 V1 已确认失败。

## 2. 当前工作区不是干净提交

观察时 `git dirty=true`。以下三项是进入 Step 1 前已经存在的工作区改动，本步不修改、不暂存：

| 文件 | 工作区 SHA-256 | 状态与解释 |
| --- | --- | --- |
| `prompts.py` | `b62feb8f53c71b1eef7a411bdd2b6d6e30c9d36437f610ec99b1d38c35ef2a85` | Day 6 实际生产评测使用的 prompt；尚未包含在 `99ac209` 中 |
| `scripts/day5_abstention_eval.py` | `a6f94504521cc778b168e9993d703b925294f943a87704037fd64770cf1936d7` | 既有 Day 5 回归脚本改动；也是 Day 6 manifest 的 protected hash |
| `knowledge/source/products/hr1060.json` | `37551d105e161c494ea12ffb78ac1cc5e61418805a88dffbec18e25769820cca` | 已把 `电池容量` 改成 `电源电压`，但尚未按知识流水线验证或传播到 documents/chunks/vectors |

因此不能把 `99ac209` 单独描述成 Day 6 实际运行时的完整生产快照；可复核基线由该提交、上表 dirty 文件哈希以及下列封存哈希共同定义。本步只提交本合同及 `.gitignore` 文档例外，不把这三项现有改动混入基线文档提交。

## 3. 基线哈希

### 生产相关文件

| 文件 | SHA-256 |
| --- | --- |
| `prompts.py` | `b62feb8f53c71b1eef7a411bdd2b6d6e30c9d36437f610ec99b1d38c35ef2a85` |
| `routes/chat.py` | `382399bc4a3702a1d376259d7497525d8eb7244116985fd18e475dee6146d37a` |
| `services/retrieval.py` | `a8064f7fa6b35a15dae9a0f8288601d652d92389d201d2e33f89a3d6eb57a71f` |
| `rag_context.py` | `0f2354de60f8d91c59e494304fd39adb1eebd584f7c164344d6e35581376f340` |
| `services/llm.py` | `d6b28d2c06ec8baececaed2e3abc8db6c6f3acd73d75b9c43aaeee2782d4f4a4` |
| `knowledge_pipeline/retrieval/entities.py` | `e608c74b9033a775df0f7554976681f9732b2b4a81e1b870e24cb3eeefa5cb59` |
| `knowledge_pipeline/retrieval/retriever.py` | `f0baae3a85aa61c53f5a61a8009c6c3818b9ba7d1869096f17827a7eef19d8b5` |

### 知识与评测封存

| 文件 | SHA-256 |
| --- | --- |
| `knowledge/documents.jsonl` | `adb239c90ce61e241e791936e79ef46b404afc1fa0582e6928ba90dca900e2c3` |
| `knowledge/chunks.jsonl` | `e39d3869d4a4be6f8f4303cc15ab2d3d5d22473442e977dc40a9bc2481c1c490` |
| `knowledge/vector_records.jsonl` | `2605623d56af69f9c0d0e639a34494f6bbdc7c2b0b3e72066e33bd3e6674aa62` |
| `eval/rag_v1.json` | `9a0d4ae927ea95d5a21e40d69db3c5ab54d353652bab20030d29dc1f4110d310` |
| `eval/rag_v1_holdout.json` | `1ba14546f9d89a47bf35ab6a2b645f09605a41291694f5284093642abfdd5cdd` |
| `eval/rag_v1_manifest.json` | `1a1072d721c56aec8b34eecef41219d3f631c7ac4b71bfbfb0c448dd3f22e9d5` |
| Dev 原始生成 | `6ace9eecd192baa1e4e0567d1223232489fc2667c153161de5935b97c2826fb4` |
| Frozen 原始生成 | `32910b6d47daf7bf997ce0904c148fc896ebcceb9abcefa6bd25991b2cd43b8b` |
| Day 6 holdout 原始生成 | `c0882d8c378806792e6f78272dae89255d3b417a299e6ec1c354fbb1413bacef` |
| Day 6 holdout 运行前冻结记录 | `098397ef5838ba559a0a7ceb6820aa90e52bd475610132d0c6cee689ddc3afbf` |

当前 review 文件 SHA-256（审核状态变更后的独立记录）：dev `a221e56ba483e6af1f74129f060f29133c67ca33a928bc62a9694ead0739728c`；frozen `99c4847706a098e88bf86482ff06bbf1da654909d974c5408e19945ad10fe47c`；holdout `0c6023e2d37a455a164e0f9ec51ded3bb459e64f43ad4032eab9bd32abf1ceb4`。历史原始 JSONL 不回写，Day 6 holdout 从本日起只称为已见回归证据，不再称为 unseen。

## 4. Day 7 验收门槛

### Blocking

- dev-027 的 retrieved-context trust-boundary failure 必须修复。即使没有输出攻击者假值，只要因攻击而隐藏、拒绝或改变合法答案，仍判失败。

### Required generation regressions

- dev-018 不得再把未记载功能推成明确不存在；已知重量已经足以作选择。
- baseline-011 必须覆盖用户要求且证据支持的网络参与者/关系，同时不要求复制无关事实。
- baseline-016 不得把只由一个来源支持的事实概括成两个来源的共同点。
- 上述失败统一归 `F — Generation Failure`；dev-027 子类为 `trust-boundary violation`。评分维度只写完整名称 Correctness、Groundedness、Completeness、Refusal，避免与 failure class G 混淆。

### Retrieval robustness

- 修复英文 exact-entity 边界缺陷；`HP780 battery`、`battery of HP780`、`information about HP780` 和独立 `HP780` 应在适用时精确识别。
- 保留 NFKC、大小写、合法空格/连字符支持及 ASCII 字母数字边界保护；不得通过删除全部边界检查造成 substring false match。
- 不因当前生成失败改变 K=5、embedding、NumPy backend 或增加 hybrid/reranking，除非后续出现新的检索证据。

### Knowledge quality

- HR1060 的 authoritative source 必须确认 `13.6V±15% / 100–240V` 的含义；若确认为供电/输入电压，通过标准流程更新 source、normalized document、chunk 和 vector record，并检查同类字段。
- 当前 dirty source 改名本身不算完成修复；Day 6 的 frozen artifacts 永不改写。

### Regression

- 已见 Dev/Frozen 案例作为 regression，不称为 unseen；四个已知失败必须逐项复核。
- 已确认的正常回答、未知拒答、公司身份、语言、防爆分类和生产协议不得发生实质退化。
- 生产 endpoint、NDJSON、request ID、超时、重试、限流、并发、输入限制、历史边界及 privacy-first logging 保持兼容。

### Final

- 所有调优冻结后，建立并人工核验新的 V1.1 holdout；冻结 ground truth、代码、prompt、知识、chunks、vectors、K、模型配置和哈希后只运行一次。
- 如果新的 holdout 暴露 blocking defect，该题集立即转为开发证据；修复后必须再建新的 unseen holdout。

### Deployment

- 只有 blocking、generation、retrieval、knowledge、regression、新 holdout、知识重建流程和 release-candidate smoke gates 全部通过后，才可标记 `AI Customer Support RAG V1.1 — Production Candidate`。
- Production Candidate 不等于 Production。没有负责人明确授权，不 push、不部署、不重启生产服务、不修改 Nginx。

## 5. Step 1 验收标准

- Day 6 manifest 内全部 artifact/protected hashes 与当前封存文件一致。
- 三次原始生成及 holdout freeze hashes 不变；三份 V1 review 均为 `complete`。
- 当前生产/dirty state 和知识快照可由本合同中的提交与哈希准确识别。
- 仅允许新增/修改本合同及其 Git ignore 例外；不得改变生产、知识、向量、eval questions 或历史结果。
- 完整离线 unittest 必须通过。
- 达成后停止，不进入 Step 2。

## 6. Step 1 验证记录

- `scripts/validate_rag_dataset.py` 离线验证通过：60题（frozen 20、dev 30、holdout 10），50题有retrieval gold，3题使用受控fixture；没有执行检索、生成或holdout调用。
- 完整离线 `unittest discover -s tests`：245项通过，0 failure、0 error，耗时24.125秒。
- Day 6 manifest 的6项artifact hashes和6项protected hashes逐项一致；三份原始生成和holdout freeze的SHA-256逐项一致。
- 三份V1独立review均为`complete`；逐题评分和历史原始运行未改写。
- `git diff --check`没有错误；只有Windows工作区现有LF→CRLF提示，不是内容失败。
- 本步没有真实provider调用、API费用、生产日志变更或部署操作。
