# Day 6 Step 3：独立检索评估入口

日期：2026-09-04。数据版本：rag-v1.0；结果schema：1.0。

后续修订：用户在Step 4开始前撤销本报告第1节所述官网添加。三条新增prompt内容及`company_website_v1`已移除，现阶段不添加联网功能。以下官网内容与prompt哈希仅为Step 3历史记录，不代表当前生效配置。

本步实现检索评估代码及测试，并运行真实本地资料的预检。**没有执行开发集、frozen或holdout的正式检索评估，没有调用付费API或DeepSeek，没有产生真实模型质量成绩。**

## 1. 本次新增的官网说明

用户确认官方网站为 `https://www.shengborun.com/`，并在澄清后选择“本步只登记官网，实际联网另行设计”。

已在system prompt登记官网，允许相关时提供网址；官网是可选参考来源，不要求每次附链接或访问。只有应用确实提供、或未来联网工具实际获取并传给模型的内容，才可以作为依据。当前没有网页访问工具，不得声称已经访问或核实官网最新内容，也不能猜造子页面、价格表或下载链接。

官网以新规则`company_website_v1`登记，既有`company_identity_v1`和`radio_classification_v1`不改。题目、期望答案和holdout标签不变；后续运行单独记录新的manifest及prompt哈希。当前prompt文件SHA-256为`6f926260c3b5095ef959cb52a78fc28c421f8350a72b454310a28a9c3902dd1e`，该prompt不参与检索计算。

### 不联网为什么能回答一般问题

模型可利用训练时学到的知识、当前对话和应用提供的上下文回答；本项目还将本地KB检索内容提供给DeepSeek。它不是每答一题都现场搜索互联网。训练知识不保证最新或准确，也不能替代公司事实依据。

当前`services/llm.py`的API调用发送messages并开启stream，没有配置网页访问工具。写入URL不会自动赋予联网能力。

### 真正联网的复杂度

固定获取某个公开页面较简单；在生产客服中按需访问官网并可靠使用内容则是中等复杂度的独立改动。至少需要域名/跳转范围限制、超时与失败处理、网页正文提取、来源和更新时间、低信任网页指令隔离，以及对隐私、延迟和评测可复现性的处理。

限定官网的按需访问可控制范围；通用搜索、任意站点、JS渲染会明显增加复杂度。这里只做工程范围判断，没有实现联网，也没有承诺工期。后续应单独确认设计，不能把当前未提供的网页内容混入Day6固定知识评测。

## 2. 实现路径

`指定split → 校验选中题集/知识/向量快照 → production build_retriever → 原Retriever.retrieve → 三个指标 → 新的受控结果文件`

- 复用`services.retrieval.build_retriever`，仍是NumPy精确cosine检索及现有exact-entity处理。
- 只将当前case的question原文传入Retriever，不重写、不加历史、不重新embedding知识块。
- 使用最终返回顺序，不按cosine重新排序：exact-entity结果可能先于分数更高的dense结果。
- 不加相似度阈值，不调K，不做rerank、BM25、hybrid、MRR或其他高级指标。
- 不调用生成服务、prompt builder或DeepSeek；官网说明与检索runner互相独立。

## 3. 指标定义

设G为期望chunk ID集合，R为返回结果前K项的chunk ID集合。

| 指标 | 定义 |
| --- | --- |
| Hit@K | G与R有交集为1，否则0。 |
| Recall@K | `len(G ∩ R) / len(G)`。多事实题只命中一部分不能算完整召回。 |
| First Relevant Rank | 最终Top-K中首个相关chunk的1-based排名；未找到记null。 |

汇总Hit和Recall为成功完成检索的有效案例的宏平均，每题权重相同；首个相关排名给出分布，未找到单独计入`not_found`。每行仍保留原始三个指标。

- 空gold：不调用embedding，`status=excluded_no_gold`、三个指标为null，明确排除分母。这包括纯未知、纯系统事实或部分系统事实题，不等于“检索失败”。
- 非空gold但无命中：Hit=0、Recall=0、首个相关排名null，是实际未命中。
- 调用错误：`status=error`，只记录error type，不记录provider异常正文。继续处理后续题，保留已完成结果。
- 存在调用错误时整次运行`status=incomplete`，命令退出码1；汇总明确`metric_scope=successfully_evaluated_cases_only`，同时报告selected/attempted/scored/excluded/error数量。不得把完成子集的成绩称为完整题集成绩。
- 没有任何可计分结果时汇总为null，不输出虚假的0%或100%。

例：需要两个chunk，只在第2名找到其中一个，则Hit@K=1、Recall@K=0.5、First Relevant Rank=2。此例为定义说明，不是真实评测结果。

## 4. 受控结果与可复现信息

每题保留case ID、分类、split、query、expected chunk/document/system-rule IDs、实际retrieved chunk IDs、指标、执行状态和error type。hit保持当前生产`RetrievalResult`的字段：rank、score、match_origin、matched_entity_ids、chunk_id、parent_document_id、type、section、text、content_hash、metadata、source_url、source_files。

这些仅属于手工编写的eval数据，不改变production logging。核心评估函数不打印query/hits；CLI输出预检数量或汇总及文件位置，不输出API密钥、workspace ID或provider异常正文。

运行元数据包括embedding provider/model/dimensions、超时/重试配置、K、数据和知识快照、manifest哈希、代码文件哈希、Git commit及dirty状态、开始/结束时间。记录prompt文件哈希供后续对照，同时明确其未用于retrieval。没有生成模型成绩或虚构的generation调用。

`fixture_id`只保留为案例关联，`fixture_applied=false`。Step3不会把合成攻击附加到chunk；因此不能凭这里的召回成绩声称成功抵御注入。

## 5. 执行保护

- 默认仅预检：读取已选题集、对应review/fixture、知识与向量，校验结构、证据和配置一致性；不构造embedding provider、不写结果、不调用API。
- 默认split为dev。frozen和holdout必须显式指定；holdout还必须显式解锁，解锁开关仅留给Step8配置冻结后使用，不会自动记录或证明配置已冻结。
- dev/frozen只读取主集文件，不加载holdout题目、review或fixture。完整题集检查仍由Step2的独立校验命令负责。
- 默认K沿用`RETRIEVAL_TOP_K`，未配置时5；`--top-k`可以显式覆盖，真实预检确认当前为5。本步没有调参。
- 输出只允许位于当前数据根目录的`eval/results/`，必须新建JSON，不覆盖输入或已有结果。默认文件名包含split、UTC时间和随机后缀。
- API执行前检查输出范围并以独占创建模式预留文件，防止并发覆盖。正常API错误会留下不完整报告；本版不提供硬中断/进程崩溃后的逐题恢复功能。

## 6. 使用方式

本次实际执行的只有预检：

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv/Lib/site-packages').Path
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/evaluate_rag_retrieval.py
```

预检结果：dev 30题，有gold 23题、排除7题，K=5，DashScope / qwen3.7-text-embedding / 1024维。`query_embedding_calls=23`表示后续执行需要处理的查询数，不是本次已发送请求数；provider自身的失败重试可能增加HTTP请求次数。

进入Step4并明确批准执行该步后，使用：

```powershell
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/evaluate_rag_retrieval.py --split dev --execute
```

可选`--top-k`和`--output eval/results/新的文件名.json`。此命令会调用付费embedding API，本步未执行；不提供一键跑全部split的默认路径。

## 7. 文件清单

| 文件 | 本步用途 |
| --- | --- |
| `evaluation/retrieval.py`（新增） | 三个指标和逐题检索评估、错误状态、宏平均与排名分布。 |
| `scripts/evaluate_rag_retrieval.py`（新增） | 预检/显式执行入口，选集、快照验证、复用生产builder、安全结果保存。 |
| `tests/test_day6_retrieval.py`（新增） | 手算指标、真实Retriever/NumPy路径、原顺序、空gold、错误及provenance测试。 |
| `tests/test_day6_retrieval_cli.py`（新增） | 临时资料上的预检/执行、holdout锁、快照漂移、输出保护与错误结果保存。 |
| `prompts.py`（修改） | 只添加用户批准的官网及访问能力边界；其他已有修改保留。 |
| `eval/rag_v1_manifest.json`（修改） | 新增独立`company_website_v1`规则，不更改case文件或其快照。 |
| `docs/day-6-evaluation-contract.md`（修改） | 记录本次新增官网规则及实际联网延期决定。 |
| `.gitignore`（修改） | 使本Step3报告可纳入版本控制，不改变results的本地保存策略。 |
| `docs/day-6-step-3-retrieval.md`（新增） | 本报告及复现说明。 |

## 8. 测试观察与边界

新增20个测试，使用真实Retriever、NumPy索引和实体解析器，仅在外部embedding边界使用确定性测试向量或响应；不构造一个永远返回预期结果的假Retriever。CLI测试只在临时目录运行并写入合成结果，不把它们当作本公司题集成绩。

测试发现一个既有实体边界细节：生产解析器先去掉空格，再检查ASCII字母数字边界。合成query `alpha parameters`不进入exact-entity路径，而`alpha 参数`可以；测试原先对前者的假设不符合当前生产行为。为单独验证“保持exact-entity顺序”，测试改用能触发现有路径的后者，未修改生产解析器。该观察记录给Step4核查英文真实问题，不据此声称已经发生公司题集检索失败，也不预先修复。

只读复核发现，provider返回`data=None`时，既有解析器抛出的TypeError会越过原有错误捕获，令已预留的结果文件为空。新增“成功→格式异常→成功”的回归测试，使用真实provider解析器与Retriever，只替换外部响应；先复现中断，再在评测逐题边界捕获普通Exception，清空失败行指标并仅保留错误类型。修复后两个成功案例均保存，异常案例排除分母，整次报告为incomplete、退出码1；不吞掉用户中断，不修改生产provider。

当前验证：新20 tests通过；完整226 tests通过；真实60题数据校验通过；真实dev预检通过；`git diff --check`通过。没有真实Hit/Recall成绩，没有live生成验证或官网联网测试。

本步结束后停止，等待明确进入Step4。
