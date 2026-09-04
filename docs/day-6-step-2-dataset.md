# Day 6 Step 2：题集扩展、隔离与证据校验

日期：2026-09-04。数据版本：`rag-v1.0`；case schema：`1.0`。

本步只建立题集、出题审查、受控攻击资料和离线校验。没有调用 embedding/DeepSeek，没有执行检索、生成或 holdout，没有产生质量分数。Step 3–8 未执行。

## 1. 题集构成

| 分类 | Frozen | Dev | Holdout | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 产品型号/参数/对比 | 7 | 10 | 3 | 20 |
| 产品推荐/选型澄清 | 1 | 3 | 2 | 6 |
| 解决方案 | 6 | 3 | 1 | 10 |
| 技术支持/服务 | 3 | 2 | 1 | 6 |
| 公司/联系/分类总览 | 1 | 5 | 1 | 7 |
| 未知问题 | 2 | 4 | 1 | 7 |
| 对抗/信任边界 | 0 | 3 | 1 | 4 |
| **合计** | **20** | **30** | **10** | **60** |

| 预期行为 | Frozen | Dev | Holdout | 合计 |
| --- | ---: | ---: | ---: | ---: |
| answer | 17 | 20 | 8 | 45 |
| partial_answer | 1 | 5 | 1 | 7 |
| abstain | 2 | 4 | 1 | 7 |
| clarify | 0 | 1 | 0 | 1 |

有检索 gold 的案例共50题：frozen 18、dev 23、holdout 9。其余10题包括纯未知、仅系统规则可答和需求澄清；以后计算检索指标时必须排除空gold，不能打成0或1。

表中是题集数量，不是准确率或评测成绩。

## 2. 分集和冻结规则

- `eval/rag_v1.json`：20条冻结基线的新版本标注＋30条开发题。冻结题ID与问题原文逐字保持，旧回答和旧评分不复制改写到这里。
- `eval/rag_v1_holdout.json`：10条新编问题，单独存储。出题时检查其证据允许；尚未提交给Retriever或生成模型。
- 旧Day5脚本中的41条不同问题只作为已用问题来源。通过AST读取字面量，不导入live脚本；holdout与这些问题以及当前数据集无规范化重复。
- 新holdout包含新的具体型号/参数组合、运行与存储温度辨析、定制配置与带外管理的局部拒答等，未拿旧调试问题换ID。领域和能力可以与dev重合；不声称是完全未见过的知识领域。
- 内容由manifest中的SHA-256固定。它用于发现意外漂移，不是防篡改签名；修改文件同时改哈希并不能证明holdout仍然未受污染。
- 未来runner必须显式选择split。开发阶段只跑dev；Step7才比较frozen，Step8才在配置冻结后跑holdout。
- 若根据holdout表现修改系统，本版holdout成为下一版开发材料，需要另建新holdout，不继续声称原题未见。

## 3. 证据与出题质量审查

两个`*_authoring.json`分别对应主集和holdout。共60条审查记录，每条包含六维三档判断、诊断目的、chunk中的原文摘录、缺失信息原因以及历史来源/标注说明。

- 六维采用用户确认的 Naturalness、Ground truth、Retrieval value、Evidence alignment、Diagnostic value、Ambiguity，不另建自动打分器。
- 精确型号题允许Retrieval value为“纯关键词”；场景题侧重语义改写；不机械要求所有题全优。
- 未知题中的Evidence alignment“不支持”表示所问事实确无支持，拒答是正确行为，不表示偷偷放入一条不可验证的正向答案。
- partial_answer将已知事实的chunk和未知部分原因分别记录。不要因价格/保修未知就忽略同题已有参数。
- `dev-025`有意缺少选型约束，Ambiguity为“很歧义”，正确行为是询问需求，不指定随意型号为gold。
- 官方英文品牌/全称以`company_identity_v1`为依据；防爆分类以`radio_classification_v1`为依据。规则单独登记，不伪装成知识库chunk。
- 对讲机分类先用真实型号资料确认身份。用户编造型号或自己添加后缀不能证明产品存在；规则不扩展到摄像机、配件，不产生认证编号或设计原因。

校验器能验证ID/父文档、摘录确实同时出现在chunk和对应document、每个gold chunk有摘录，以及未知题填写了理由。**它不能自动证明语义蕴含，也不能仅凭一个理由字符串证明全库不存在事实。**本次已由助手逐题对照证据和边界审核，仍是可供用户复核的标注，不称为用户已完成人工审核，也没有生成评分。

未知项审核结合当前61份document/73个chunk、系统规则和应用能力；对价格、库存、保修、收入、资费等核对全库及网站产品/服务/公司内容，对具体问题再检查对应产品/服务材料。没有用检索低分证明未知，也没有通过联网补充公司知识或用行业常识补齐答案。

## 4. 历史标注修订和已发现的数据问题

| 案例 | 新标注与原因 | 原件处理 |
| --- | --- | --- |
| baseline-004 | HP780明确非防爆；HP790Ex防爆，参数分别绑定。依据用户确认的全对讲机规则，移除旧“不得声称HP780不防爆”。 | V0不变 |
| baseline-017 | 推荐优先进一步确认HP790Ex，明确HP780非防爆；不能扩展为任何危险区域都适用。 | V0不变 |
| baseline-008 | 只要求部门协作和紧急优先相关功能，不强制补充未问的架构、录音或远程调度。直接使用系统功能chunk。 | V0不变 |
| baseline-019 | 保留参考距离和不能保证任意地形的边界；不再要求额外解释KB未给出的环境影响机制。 | V0不变 |
| baseline-005 | HR1060源网站把供电电压写在“电池容量”字段；原文和chunk均保留此缺陷。ground truth按V/AC/DC语义识别供电，不称为mAh。 | 网站、document、chunk不变 |
| baseline-007 | PNE380原资料关于“最多16节点，可进一步扩展”的句子不够规范。按资料限定陈述，不外推更大数值，最长待机不当作持续工作保证。 | 资料不变 |

以上是标注与源资料检查，不是运行失败归因结果；未执行Step4或Step6。V0/V1未来应在同一Day6口径下另存复评分，同时保留旧评分并说明规则差异，不能将全部差异归功于RAG。

证据集合尽量最小且有明确职责。例如企业容量题只标正文（含30/60终端示例）；E-center多参数题只标specifications，不把只有“便携/三屏”的overview算成完整答案。多产品比较保留两个产品的必要chunk。

## 5. 受控检索内容攻击

3份合成fixture：dev有两份，holdout一份，分别测试篡改参数指令、伪造系统角色并替换收件邮箱、伪装评测授权篡改数量。另有1条dev用户输入层的管理员冒充题。

- 合成内容使用独立`eval/fixtures/`文件；攻击邮箱使用保留的`.invalid`域名，没有真实外传目标。
- `fixture_id`只指向测试数据，当前不执行。未来仅在受控评估副本中给目标命中chunk追加文本，不改变源chunk、原始向量或生产prompt。
- 保留干净事实gold。未来若Retriever未召回攻击目标，应记“攻击未执行/未覆盖”，不能声称成功抵抗攻击。
- 合成上下文结果单独标识，不能混为未经干预的production端到端成绩，也不向production logs写入任何测试查询或hit。

## 6. 新增/修改文件

| 文件 | 用途 |
| --- | --- |
| `evaluation/__init__.py` | 标记独立评测工具包，不进入生产请求链路。 |
| `evaluation/dataset.py` | case结构、引用/证据校验、快照和分集检查；无指标、provider或模型调用。 |
| `scripts/validate_rag_dataset.py` | 离线校验入口；成功输出数量，失败返回非零退出码。 |
| `tests/test_rag_dataset.py` | 使用临时合成资料验证校验逻辑、错误分支和命令入口，不依赖本机历史V0/向量文件。 |
| `eval/rag_v1.json` | 20 frozen＋30 dev的新标注。 |
| `eval/rag_v1_holdout.json` | 10条独立存储的holdout题。 |
| `eval/rag_v1_authoring.json` | 主集50题的六维审查、证据摘录及缺失信息说明。 |
| `eval/rag_v1_holdout_authoring.json` | holdout10题的独立审查记录。 |
| `eval/fixtures/rag_v1_attacks.json` | 两份开发集检索内容攻击资料。 |
| `eval/fixtures/rag_v1_holdout_attacks.json` | 一份未执行的holdout攻击资料。 |
| `eval/rag_v1_manifest.json` | 数据版本、快照哈希、规则来源、评分/验收/冻结政策。 |
| `docs/day-6-step-2-plan.md` | 本步限定范围、文件划分、测试和执行计划。 |
| `docs/day-6-step-2-dataset.md` | 本报告。 |
| `docs/day-6-evaluation-contract.md`（修改） | 补记已批准的题量、评分责任、N/A及验收决策。 |
| `.gitignore`（修改） | 仅为上述Day6题集、fixture和三份文档开定向例外；历史V0和旧结果策略不变。 |
| `.gitattributes`（修改） | 固定新题集、manifest、fixture及被快照引用的旧问题脚本为LF，避免Windows换行转换破坏哈希；不重写任何V0文件。 |

没有修改生产prompt、Retriever、Context Builder、API、日志、前端或知识内容。工作区已有的Day5修改/旧结果删除保持原状；本步不重复清理、不提交Git。

## 7. 复现和边界

```powershell
$env:PYTHONPATH=(Resolve-Path '.venv/Lib/site-packages').Path
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' scripts/validate_rag_dataset.py
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest tests.test_rag_dataset
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest discover -s tests
```

真实题集校验需要本机保留的V0两份原始文件、Day3知识和Day4向量快照；迁移工作区时需安全转移这些历史文件，不重新生成V0。单元测试使用独立临时资料，没有这一依赖。校验仅读取快照/内容，不读取`.env`，不请求网络。

校验覆盖：60题、20/30/10分集、7分类、ID与规范化问题唯一、20条原始问题与回答保留、文档/块引用和父子关系、规则ID、60条审查、证据摘录、缺失信息理由、3份合成fixture、41条历史Day5问题去重、文件哈希。

最终实测结果：

- `scripts/validate_rag_dataset.py`：退出码0，60题与所有快照/引用检查通过。
- `python -m unittest tests.test_rag_dataset`：23 tests，OK。
- `python -m unittest discover -s tests`：206 tests，OK；本步新增23个测试。
- `git diff --check`：通过，只有未设LF策略的其他旧文件换行提示，无空白错误。
- 新增8个LF属性逐项检查通过；Day6输入不再被忽略，V0、未来results和其他旧文档的忽略规则保持。
- V0两文件、documents、chunks、vector_records的SHA-256与Step1及本步开始值一致；生产prompt未修改，当前哈希为`b62feb8f53c71b1eef7a411bdd2b6d6e30c9d36437f610ec99b1d38c35ef2a85`。
- 没有暂存或提交文件；没有调用任何付费API。

以上是离线校验与软件测试结果。没有Hit@K、Recall@K、生成分数或holdout成绩；这些属于后续步骤。Step 2完成，在这里停止。

## 8. 收尾复核

只读审查未发现严重语义标注或类别覆盖问题。已处理两项反馈：为新题集及已引用的Day5问题脚本固定LF换行；补足dev-021关于高清视频回传指挥所的证据摘录，并在本步冻结完成前更新该审查文件的哈希。holdout题目、答案、证据和fixture均未因模型表现修改，也未运行。

测试方面，将bundle测试改为临时合成数据，避免依赖本机未跟踪的V0原件和向量文件；新增命令行成功/失败退出码测试。真实题集仍由独立离线校验验证，不把合成测试资料作为真实题集的正确性证据。
