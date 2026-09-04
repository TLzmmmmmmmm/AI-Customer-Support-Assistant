# Day 6 Step 1：具体清理清单（已批准并完成）

本清单只处理 docs/、eval/ 的旧测试输出。当前评测依据统一转到 Day 6；原始 V0 baseline 是明确例外，完整保留。

## 保留

- **eval/baseline_v0.json**：20 条冻结问题、参考答案、原始 V0 回答和逐题历史分数；禁止改写。
- **eval/baseline_v0_results.json**：V0 原始汇总与运行元数据；禁止改写。
- eval/retrieval_v1.json、eval/retrieval_v1_1.json：旧题集与 chunk 标注，不是运行结果；现有测试依赖，且 Step 2 可参考。
- docs 的知识/切块/检索 schema、架构、superpowers/specs 与 plans：当前实现与教学参考，不因日期旧而删除。
- docs/day-5-alpha-acceptance.md：保留 Day 5 验收记录，已注明用户暂定通过及新的评测口径。
- docs/day-5-partial-abstention.md、day-5-language-referral-fix.md、day-5-company-identity.md、day-5-radio-policy.md：保留简要历史实验与失败归因线索，不作为当前结果；清理后标注原始输出已移除，避免读者误以为链接仍有效。
- docs/day-6-evaluation-contract.md 与本清单：当前 Step 1 产物。
- scripts/ 和 tests/：不批量删除；旧案例脚本留作开发题来源，旧标签必须按 Day 6 规则重新审核。

V0 的两个文件目前被 Git 忽略且未跟踪；尤其不能假设它们可以从 Git 恢复。已记录 SHA-256，清理前后必须一致。

## 已移入回收站的 42 个文件

已按用户批准移除以下显式路径，没有按目录或未来新增通配符删除。合计 2,465,362 字节（约 2.35 MiB）。它们是旧运行输出，不是题集、源文档、chunk 或向量。

- `docs/day-5-abstention-baseline-1.jsonl`
- `docs/day-5-abstention-baseline-2.jsonl`
- `docs/day-5-abstention-v1-1.jsonl`
- `docs/day-5-abstention-v2-1.jsonl`
- `docs/day-5-abstention-v2-2.jsonl`
- `docs/day-5-abstention-v3-1.jsonl`
- `docs/day-5-abstention-v3-2.jsonl`
- `docs/day-5-abstention-v3-3.jsonl`
- `docs/day-5-abstention-v3-holdout-1.jsonl`
- `docs/day-5-abstention-v3-holdout-2.jsonl`
- `docs/day-5-company-grounding-regression.jsonl`
- `docs/day-5-company-identity-baseline-network.jsonl`
- `docs/day-5-company-identity-baseline.jsonl`
- `docs/day-5-company-identity-confirmed.jsonl`
- `docs/day-5-company-language-regression.jsonl`
- `docs/day-5-language-baseline.jsonl`
- `docs/day-5-language-l1.jsonl`
- `docs/day-5-language-l2.jsonl`
- `docs/day-5-language-q1-focus-1.jsonl`
- `docs/day-5-language-q1-focus-2.jsonl`
- `docs/day-5-language-q1-holdout.jsonl`
- `docs/day-5-language-q1-regression.jsonl`
- `docs/day-5-language-q1-resources.jsonl`
- `docs/day-5-language-q2-focus-1.jsonl`
- `docs/day-5-language-q2-holdout.jsonl`
- `docs/day-5-language-q2-regression-1.jsonl`
- `docs/day-5-language-q2-regression-2.jsonl`
- `docs/day-5-language-q2-resources.jsonl`
- `docs/day-5-language-q3-focus.jsonl`
- `docs/day-5-language-q3-resources-1.jsonl`
- `docs/day-5-live-smoke.jsonl`
- `docs/day-5-radio-policy-baseline.jsonl`
- `docs/day-5-radio-policy-confirmed.jsonl`
- `docs/day-5-radio-policy-holdout.jsonl`
- `docs/day-5-radio-policy-identity.jsonl`
- `docs/day-5-radio-policy-r2-confirmed.jsonl`
- `docs/day-5-radio-policy-r2-holdout.jsonl`
- `docs/day-5-radio-policy-r2-identity.jsonl`
- `docs/day-5-radio-policy-r2-regression.jsonl`
- `docs/day-5-radio-policy-regression.jsonl`
- `eval/retrieval_v1_results.json`
- `eval/retrieval_v1_1_results.json`

文件名中的 day-5-...baseline 表示 Day 5 的实验对照，不是需要保留的 Day 1 V0 baseline；两者已分别核实。

## 已批准的执行要求（保留供核对）

1. 用户确认这份具体清单后才执行。再次解析绝对路径，确认全部位于项目 docs/ 或 eval/ 下，且不包含 V0 文件、目录或符号链接；逐个检查目标，避免越界。
2. 13 个目标已被 Git 跟踪，29 个未跟踪。优先移到 Windows 回收站，保留可恢复性；如果回收站操作不可用，停止并说明，不悄悄改为永久删除。移动到回收站后仅在尚未清空且系统允许恢复时可恢复；Git 也不能恢复未跟踪版本。
3. tests/test_retrieval_evaluation.py 中 test_historical_v1_suite_and_result_are_preserved 直接断言旧结果文件存在。清理时把它改为仅检查保留的历史题集，并移除不再使用的结果路径常量；不删除/放宽检索行为、ground truth、快照或 CLI 功能测试。
4. 给保留的历史实验文档增加统一清理说明，标明旧 raw output 已按本清单移除、旧分数不是当前验收依据。历史设计/实施计划中提及旧输出文件的内容保留为历史描述，不伪造当时结果。
5. 默认输出路径仍出现在旧 CLI 中仅表示“下次运行写到哪里”，不是启动依赖；Step 1 不运行这些付费脚本，不把输出重新生成。Day 6 的结果目录约定为 eval/results/，由后续 runner 实现。
6. 执行后核对 42 个目标状态、V0 两文件哈希、知识/向量未变，并运行相关测试；报告删除和恢复方式，停止，不进入 Step 2。

## 当前状态

用户已明确批准本清单。2026-09-03 执行了 42 次 SendToRecycleBin，全部成功，原路径均不存在；没有使用永久删除回退。随后在用户回收站中按名称和原目录核对，找到了全部 42 个唯一文件。未清空回收站；在回收站内容仍保留时可通过 Windows 回收站恢复。

V0 两文件在操作前后与 Step 1 初始 SHA-256 完全一致。现有题集、知识文档、chunk 和向量保留。回收站在普通受限环境中不可见，复核使用与移动操作相同的授权环境；这不表示文件被永久删除。

先运行旧存档测试，确认唯一失败点为已删除结果文件的存在断言；随后按批准改为 test_historical_v1_suite_is_preserved，仅保留历史题集存在检查，并移除不用的结果路径常量。未删减检索功能测试。五份 Day 5 历史实验文档已增加统一清理说明。

最终复测：检索评测模块 10 项通过，全套 183 项通过；V0、documents、chunks、vector_records 五个文件哈希均未变。docs/ 与 eval/ 保留 30 个文件，42 个目标全已移出。详情见 [Step 1 契约记录](day-6-evaluation-contract.md)。本次未运行付费 API，也未进入 Step 2。
