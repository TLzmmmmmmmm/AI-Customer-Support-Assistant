# Day 6 Step 4：开发集检索结果与诊断

日期：2026-09-04。仅执行dev检索评测，停止于Step 4。

## 1. 结论与边界

真实DashScope embedding API、现有production Retriever和NumPy精确cosine索引下，dev的23个有检索gold案例均完整召回：Hit@5=100%，宏平均Recall@5=100%。30题中7题无检索gold，排除指标分母，不算成功或失败。

没有发现本次dev的Top-5漏召回；不代表最终答案正确、不代表拒答或抗注入通过，也不代表整个Day 6或holdout通过。本步不生成回答、不修改K、知识、标注、检索算法、生产日志或前端。

## 2. 官网添加撤销

按用户最新要求，删除Step 3新增的三条官网prompt内容和manifest中的`company_website_v1`。当前明确不添加联网功能。Step 3报告保留历史描述，并加上已撤销标记；当前契约已同步。

`prompts.py`恢复到官网添加前的SHA-256：`b62feb8f53c71b1eef7a411bdd2b6d6e30c9d36437f610ec99b1d38c35ef2a85`。没有撤销更早的Day 5规则，也没有移除本地知识中既有source_url等provenance。那些URL是资料来源记录，不是实时访问功能。

## 3. 两次运行分别保留

| 运行 | 状态 | 有gold查询 | 成功计分 | 错误 | 指标用途 |
| --- | --- | ---: | ---: | ---: | --- |
| `retrieval-dev-20260904T085055Z-32477a0b.json` | incomplete | 23 | 0 | 23 | 环境/API执行错误记录，指标为null |
| `retrieval-dev-20260904T085153Z-1b04eb5b.json` | completed | 23 | 23 | 0 | 本报告的正式dev检索成绩 |

原始逐题结果均位于`eval/results/`，按现有策略本地保存，不覆盖旧文件、不写入production日志。首次受限环境运行每个有gold案例均记录EmbeddingAPIError；放开沙箱网络后，同一配置重跑全部成功，支持环境访问限制为原因。错误正文按隐私策略未保存，因此不声称已确定底层DNS/代理/HTTP错误细节。不能将首次运行写成Hit@5=0%，也不能把重试当作修改系统后的效果提升。

重跑run_id：`cdc4152758b342889366839513b1787e`。开始时间见文件名UTC时间（北京时间16:51:53）；具体起止时间在原始结果中。Embedding为dashscope / qwen3.7-text-embedding / 1024维，K=5，timeout=30秒、max_retries=2。知识向量全部复用，没有重新embedding知识块。报告中的23是有gold查询数，不是已核算的HTTP请求次数或账单token数；未提供实际费用数字。

运行记录包含题集/知识快照、manifest及代码哈希，Git commit为`bceec4f8f84d6e44b45161a8031b0f9c1c10c701`、dirty=true。prompt哈希也记录，但prompt不参与检索计算。

## 4. 指标与分类分布

- Hit@5：23/23=100%。
- Recall@5：23个案例的宏平均为100%，包括4个双chunk案例；没有只召回部分gold的案例。
- First Relevant Rank：第1名22题，第3名1题；其他排名及未命中均为0。保留最终Retriever顺序，不按score重新排序。
- 检索执行错误：成功重跑为0；首轮23个执行错误单列，不混入未命中分类。

| 类别 | dev总数 | 计分 | 排除 | Hit@5 | Recall@5 | 未完整召回 |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| product_spec | 10 | 10 | 0 | 100% | 100% | 0 |
| product_recommendation | 3 | 2 | 1 | 100% | 100% | 0 |
| solution | 3 | 3 | 0 | 100% | 100% | 0 |
| support | 2 | 2 | 0 | 100% | 100% | 0 |
| company | 5 | 3 | 2 | 100% | 100% | 0 |
| unknown | 4 | 0 | 4 | N/A | N/A | N/A |
| adversarial | 3 | 3 | 0 | 100% | 100% | 0 |

排除案例：dev-010、011、012、014、024、025、029。它们的未知事项、系统身份或澄清行为要在后续生成评测中检验，本次没有发送其query embedding。

dev-005、006、009、018的两个期望chunk均命中。14个计分案例使用了exact_entity路径，9个只使用dense路径；这是实际结果路径描述，不是独立dense模型的消融成绩。不能据此宣称纯dense对全部23题也会达到100%。

## 5. 逐项诊断

### 5.1 本次未发生的失败类别

以本次有gold案例为限，未完整召回案例为0，所以没有可归入Knowledge Missing、Ingestion / Schema Failure、Chunking Failure或Retrieval Failure的失败案例。无gold题被有意排除，不能标成Knowledge Missing故障。并未据此证明整库没有任何采集或分块问题。

### 5.2 dev-019：正确证据第3名，不是Top-5失败

问题：我们几个部门不在同一个地方，平时只想听本组的消息，必要时又要跨区联系，你们的企事业单位方案怎么做到？

| 排名 | chunk_id | cosine score | 作用 |
| --- | --- | ---: | --- |
| 1 | `solution:enterprise:body:industry-background` | 0.718970 | 描述部门互通和分组困难，匹配用户问题情境，但不是解决机制依据 |
| 2 | `solution:enterprise:overview` | 0.714956 | 摘要中提及通话分组和IP跨区互联，未展开具体机制 |
| 3 | `solution:enterprise:body:solution` | 0.685485 | gold：按工作性质通话群分组、通过IP/因特网连接远处部门 |
| 4 | `solution:hotel:overview` | 0.648102 | 相邻行业的分组与IP互联内容，不替代本题gold |
| 5 | `solution:petrochemical:body:solution` | 0.632220 | 相邻行业方案，不替代本题gold |

证据链核查：

1. 权威本地源`D:/Shengborun/src/content/solutions/enterprise.md`的“解决方案”第2、3条确有分组与跨区互联机制；没有访问网站。
2. `knowledge/documents.jsonl`中`solution:enterprise`保留相同内容。
3. `knowledge/chunks.jsonl`中gold章节存在；运行前校验向量记录与chunks一致。
4. authoring review引用的三处机制证据均来自该gold，选择详细章节合理，不因结果修改标签。
5. Retriever实际第3名返回完整gold，Hit=1、Recall=1。未发生知识缺失、采集丢失、分块丢失或Top-5漏召回。

排名现象的解释：dense cosine衡量文本相似性，不直接判断哪一段最完整地回答问题；“行业背景”的问题描述与用户措辞接近，因此可能排在解决机制之前。该解释基于文本与分数，是排序现象分析，不是对embedding内部机制的证明。当前正确依据已进入Top-5，建议不为这一题调K、改prompt或引入reranker；后续检查生成是否正确使用第3条依据。

### 5.3 英文实体边界：已复现的潜在风险，未计入本次失败数

使用真实73条vector records构建ExactEntityResolver，仅做本地解析，不调用embedding或generation：

| 独立诊断输入（非正式dev新增题） | 解析出的型号 |
| --- | --- |
| `HP780的防护等级是什么？` | `product:hp780` |
| `What is the ingress protection rating of HP780?` | 无 |
| `What is the HP780 ingress protection rating?` | 无 |

原因：`entities.py`先删除空格/连字符，再检查ASCII字母数字边界。英文单词与型号被拼接，例如`ofHP780`，使正式型号不能通过边界检查。中文邻接字符不属于ASCII字母数字，因此不触发相同问题。

此问题属于exact-entity解析层潜在Retrieval Failure风险；dense兜底仍可能正确找回，不应仅凭解析未匹配就声称最终召回失败。本次dev唯一计分的英文问题dev-013问公司联系方式，没有这个型号边界情境；dev-029虽含HP500，但无检索gold而未执行。因此dev全通过未覆盖该风险。

建议另行批准一个最小实体边界修复：在保留词边界的前提下处理型号内部空格/连字符；验证自然英文、中文邻接、CQST/Ex、带空格型号，以及防止HP780错误命中HP780CQST/虚构前后缀。先用离线解析回归，再用独立开发诊断确认检索表现，不改冻结/holdout或回写本次成绩。本步没有实施该修复，也没有修改生产代码来绕过问题。

## 6. 限制与下一步建议

- 样本只有23个计分案例，部分来自既有Day 5开发问题；不能等同未见数据上的泛化成绩。
- 完整召回不保证context builder保留证据，也不保证DeepSeek正确使用，特别是局部拒答、因果推测、英文和重复总结。
- 未评估Top-5无关内容比例，不新增Precision等指标；方案题出现其他行业内容，应在生成步骤检查是否混用。
- adversarial的3题此处只证明干净知识被找回；没有应用检索内容攻击fixture，没有检验泄露或指令服从。
- 建议保持K=5及现有检索架构；英文实体边界修复待单独批准。Step 5需用户明确指示后才执行。

## 7. 文件与验证

修改：`prompts.py`、`eval/rag_v1_manifest.json`撤销官网添加；`docs/day-6-evaluation-contract.md`同步当前范围/进度；`docs/day-6-step-3-retrieval.md`标记历史撤销；`.gitignore`定向允许本报告。

新增：本报告，以及`eval/results/`中上述两份独立运行记录。没有修改任何runner、检索实现、测试代码或数据集标签；原有source URL和V0文件保留。不提交Git。

验证：

- 正式执行前dev预检通过，30题/23次计划查询/7排除，K=5。
- 沙箱外正式重跑completed，23个计分案例、0执行错误；原始逐题结果及完整provenance已保存。
- 全套unittest：226 tests，OK（本步未新增功能代码或测试方法）。
- manifest中全部artifact/protected SHA-256逐项匹配，包含V0、知识、向量和未执行holdout文件；只核对holdout文件字节哈希，不读取其题目进行本步分析。
- prompt哈希与Step 3添加官网之前完全一致；官网URL和company_website_v1均已从相应生效配置移除。
- `git diff --check`通过。生产接口、日志、联网能力不变；未运行frozen、holdout或任何DeepSeek生成。

评估工程：Step 4完成。模型整体质量验收：尚不能据此判定；失败分类与建议由助手基于证据提出，未冒称用户已审核。停止，不进入Step 5。
