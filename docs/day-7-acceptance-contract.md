# Week 2 Day 7 Revised：RAG V1.1 Acceptance Contract

状态：Revised Step 1 验收范围已冻结；Step 2 尚未开始。本合同取代旧版 Day 7 blocking list，不改写 Day 6 原始生成结果。

## 1. 目标与执行约束

Day 7 的目标是将 Day 6 RAG V1 稳定为 `AI Customer Support RAG V1.1 — Production Candidate`，而不是部署到生产。每次只执行一个 revised numbered step，报告后停止，等待 owner 授权下一步。

保持生产架构：Astro → `POST /api/chat-stream` → FastAPI → Retriever → Context Builder → DeepSeek streaming → `application/x-ndjson`。事件类型仍为 `delta` / `done` / `error`，并保留 X-Request-ID、timeout、retry、rate limit、concurrency/input/history 边界和 privacy-first production logging。

冻结配置：DeepSeek `deepseek-v4-flash`；DashScope `qwen3.7-text-embedding`、1024 维；Exact entity + NumPy exact cosine；Top-K=5。没有新证据与 owner 批准时，不增加 reranking、BM25/hybrid、vector database、query rewriting、similarity threshold、web search、LLM judge、claim checker、visible citations 或 production retrieval telemetry。

## 2. Revised owner acceptance decisions

| 案例/风险 | 当前处置 | V1.1 要求 |
| --- | --- | --- |
| `dev-018` | PASS | 不需修复；按下方有限 closed-world 规则回归 |
| `baseline-011` | PASS / owner accepted | 不需修复 |
| `baseline-016` | PARTIAL PASS / non-blocking / low priority | 保留局限；除非其他变更造成回归，Day 7 不优化 |
| `dev-027` synthetic retrieved-context attack | Known future-hardening limitation / non-blocking | 保留证据；当前不实现 retrieved-context prompt-injection hardening |
| English ExactEntityResolver boundary | Known low-impact limitation / non-blocking | 不修改 entity matching；miss 时继续由 dense retrieval fallback |
| HR1060 source/schema | BLOCKING | 必须通过 source → documents → chunks → embeddings/vectors 标准流水线修复 |

已批准保留的先前变更：用户消息 prompt-injection 测试证据，以及“只有全英文问题必须全英文”的 RAG 英文外层说明修复。新版的“Day 7 不实施 prompt-injection hardening”仅指 `dev-027` 的 retrieved-context 间接注入，不撤销上述已批准内容。

## 3. Closed-world 业务语义

仅对公司权威 source 中明确作为完整枚举维护的产品 `features` / “产品特点、功能列表”字段使用：

```text
feature not recorded
→
feature treated as not supported
```

自由文本、产品介绍或技术参数中偶然未提及的能力，不因沉默自动变为不支持。价格、库存、保修、认证、商务条款、SLA 和动态运营状态始终使用 open-world 语义：未记录即 unknown。

## 4. 冻结起点

### Python RAG repository

- 分支：`week2-RAG`
- Revised Step 1 的 parent commit：`3618e3c9a2073a0d8a5cbee79984d527b7a0d18b`
- 工作区：dirty；包含已批准的先前变更、Step 1 metadata 和待 Step 2 处理的 HR1060 source，不能用 parent commit 单独重建完整候选状态。

关键先前 dirty/untracked 文件哈希：

| 文件 | SHA-256 | 含义 |
| --- | --- | --- |
| `prompts.py` | `44df09f8b6dbbd67743bd1313585a06c97897aa8f920aa363d4acddc826fe664` | 已批准 prompt 与全英文 RAG envelope 变更；本步不修改 |
| `tests/test_prompts.py` | `490edf8bad32e114ebd355e397fc44c695c774baf9889609be51fd86f8667fb6` | 已批准语言边界测试；本步不修改 |
| `scripts/day5_abstention_eval.py` | `a6f94504521cc778b168e9993d703b925294f943a87704037fd64770cf1936d7` | 早期已存在的 Day 5 变更；本步不修改 |
| `knowledge/source/products/hr1060.json` | `37551d105e161c494ea12ffb78ac1cc5e61418805a88dffbec18e25769820cca` | owner 已把字段改为电源电压；流水线尚未重建 |
| `eval/user_prompt_injection_v1.json` | `e1e66e1edac785e88161314b10beb6af22c2129b166ea3c1d53b053537a206c5` | 已批准用户消息注入回归集 |
| `scripts/evaluate_day7_hardening.py` | `a8e177d311bb9eff685c9ad3e841c13f673c9e9732524bb61701c5f1fdfee882` | 已有受控实验 runner |
| `evaluation/trust_boundary.py` | `388ce9230643197b28eaa80723420ad6a2b18469576d06c5c3ed0888ac99fa0b` | 已有实验契约/账本工具 |
| `scripts/evaluate_trust_boundary.py` | `0e7af4979740b80e9f330b7fbe35a5e34db5a4b94d3bfd7f72f05a39c7da951a` | 已有 Step 2 研究 runner |
| `tests/test_day7_trust_boundary.py` | `a8aabee625300945677652e5859bb16d1d9964f0c1d224ccf012b92c50724e3a` | 已有评测基础设施测试 |

### Astro authoritative source repository

- 分支：`AI-Assistant-Develop`
- commit：`77dfdbcc6861a407db08ee625358921cd421878e`
- `src/content/products/two-way-radio/hr1060.json` 为 dirty，SHA-256 `569069e234d9abed0370f4d444336017fa3b9e678e1fbb4cfcbdf234143dba50`。
- owner 已确认两个本地 source 中的 `13.6V±15% / 100–240V` 代表电源/输入电压。Revised Step 2 以 Astro 与 Python source 语义一致作为权威依据，不访问网站。

## 5. Day 6 immutable evidence

| 文件 | SHA-256 |
| --- | --- |
| `knowledge/documents.jsonl` | `adb239c90ce61e241e791936e79ef46b404afc1fa0582e6928ba90dca900e2c3` |
| `knowledge/chunks.jsonl` | `e39d3869d4a4be6f8f4303cc15ab2d3d5d22473442e977dc40a9bc2481c1c490` |
| `knowledge/vector_records.jsonl` | `2605623d56af69f9c0d0e639a34494f6bbdc7c2b0b3e72066e33bd3e6674aa62` |
| `eval/rag_v1.json` | `9a0d4ae927ea95d5a21e40d69db3c5ab54d353652bab20030d29dc1f4110d310` |
| `eval/rag_v1_holdout.json` | `1ba14546f9d89a47bf35ab6a2b645f09605a41291694f5284093642abfdd5cdd` |
| `eval/rag_v1_manifest.json` | `1a1072d721c56aec8b34eecef41219d3f631c7ac4b71bfbfb0c448dd3f22e9d5` |
| Dev raw generation | `6ace9eecd192baa1e4e0567d1223232489fc2667c153161de5935b97c2826fb4` |
| Frozen raw generation | `32910b6d47daf7bf997ce0904c148fc896ebcceb9abcefa6bd25991b2cd43b8b` |
| Day 6 holdout raw generation | `c0882d8c378806792e6f78272dae89255d3b417a299e6ec1c354fbb1413bacef` |
| Day 6 holdout freeze | `098397ef5838ba559a0a7ceb6820aa90e52bd475610132d0c6cee689ddc3afbf` |

Day 6 的 dev/frozen/holdout review 状态均为 `complete`。当前 review 通过独立 metadata 记录 owner 的重新处置；不改写上表 raw JSONL。Day 6 holdout 已经是 seen historical regression evidence，不得再称为 unseen。

已保留的关键后续研究证据：`trust-boundary-initial_pairs...jsonl` SHA-256 `d69d1ed38aa7d7d1d72a89e75187cc81fe7e102824f1c41f751c03362adccda7`；`trust-boundary-diagnostic_repeats...jsonl` SHA-256 `29c562c4d6e237bb27ad5d0c31910319fca3cc3fd658035fd84f51e32ea2fca9`；用户消息注入初测/英文修复前/修复后证据 SHA-256 依次为 `e05df148735d3ac26c181fa86ce2478d22e5264b4b38eeaa7bfb6fc72402d501`、`646db43d51fca1da31ce40ebddc122a3b91e76a0b78cf98f7b2529581a3e4277`、`9af1ed1da784da37c85c77a3bedd4b31a0831f5401720e92d6d7a4acf537706a`。

## 6. Revised Day 7 quality gates

1. Revised Step 2：HR1060 语义必须从两份权威 source 一致传播到 documents、chunks 和 vectors；不得手工修补生成产物，不得改写 Day 6 frozen artifacts。
2. Revised Step 3：形成 fail-closed 的全量知识重建和服务器 vector artifact 产生/加载流程；仅文档化未来 content-hash 增量路径。
3. Revised Step 4：Dev/Frozen 作为 seen regression，记录检索与生成延迟；无证据时明确延后 Top-K 实验。
4. Revised Step 5：所有调优冻结后，先人工完成 12–15 题 V1.1 holdout ground truth，再封存并只运行一次。
5. Revised Step 6：仅在本地/开发环境进行 release-candidate smoke，包括 HR1060、NDJSON、request ID、错误、资源限制与隐私日志；不部署、不 push、不重启生产服务、不修改 Nginx。

最终 Known Limitations 必须保留 `baseline-016` 精度局限、English ExactEntityResolver boundary 和 retrieved-context indirect prompt-injection limitation，并列出所有未引入的高级功能。

## 7. Revised Step 1 acceptance

- 只允许更新验收合同、review 处置和相关历史文档注记；不改 production behavior、source、documents、chunks、vectors、eval questions 或 raw results。
- Day 6 manifest 和 raw hashes 必须保持不变；三份 V1 review 状态必须为 `complete`。
- 正常离线数据集验证、完整 unittest 和 `git diff --check` 必须通过。
- 本步可选择性提交 metadata/documentation，但不得将上述先前 production、HR1060 或研究代码/原始结果混入该 commit。

## 8. Verification record

- `scripts/validate_rag_dataset.py`：通过。60 题（frozen 20 / dev 30 / Day 6 historical holdout 10）、50 题有 retrieval gold、3 题有受控 fixture；未执行 retrieval、generation 或 provider 请求。
- 完整 `.venv\\Scripts\\python.exe -m unittest discover -s tests`：255/255 通过，0 failure、0 error，耗时 22.058 秒。测试日志中的 422/429/5xx/timeout 是既有错误路径的预期仿真，不是外部 API 故障。
- Day 6 manifest 验证通过；documents、chunks、vectors、三份 raw generation 和 holdout freeze 的 SHA-256 与本合同记录一致。
- dev/frozen/holdout 三份 V1 review 均为 `complete`；owner disposition 只写入独立 review/metadata，raw JSONL 未修改。
- `git diff --check`：退出码 0；只有 Windows 工作区 LF→CRLF 提示，无 whitespace error。
- Revised Step 1 没有修改 production code、HR1060 source、documents、chunks、vectors、eval questions 或原始实验结果，也没有进行任何付费 API 调用。

结论：Revised Step 1 PASS。HR1060 全链路修复仍是唯一已确认 blocking 修复，必须等 owner 授权后在 Revised Step 2 单独执行。
