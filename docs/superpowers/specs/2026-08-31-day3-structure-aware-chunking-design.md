# Day 3 Structure-Aware Chunking V1 Design

**状态：** 对话设计已确认，等待书面 spec 审阅
**日期：** 2026-08-31
**范围边界：** 把 Day 2 已验证的 Normalized Documents 转换为确定、可验证的 Chunks，并严格停在 embedding 之前。

## 1. 目标

在现有 Day 2 knowledge pipeline 之上增加生产级 structure-aware chunking：

    knowledge/documents.jsonl
            ↓ Input Validation
    Type-aware Semantic Parsing
            ↓ Structure-aware Chunking
    Chunk Validation + Coverage Validation
            ↓ Atomic Deterministic Serialization
    knowledge/chunks.jsonl

本阶段只实现一种生产策略：`structure_aware`。已明确删除 `whole_document`、`fixed_size` 以及实验和评估框架。

本阶段不包含 embedding、tokenizer、vector database、similarity search、semantic retrieval、reranking、LLM call 或 RAG generation。

## 2. 输入边界

Day 3 只读取 `knowledge/documents.jsonl`，不读取 `knowledge/source`、Astro Content 或网站仓库。

这保持了清晰的流水线边界：

    knowledge/source
        → Day 2 Normalization
        → knowledge/documents.jsonl
        → Day 3 Chunking
        → knowledge/chunks.jsonl

`documents.jsonl` 当前全部 60 个 Document 均视为已发布知识。Day 3 不增加 draft/published 过滤；每个输入 Document 必须至少生成一个 Chunk。

## 3. 正式交付物

    knowledge/
    └── chunks.jsonl

    knowledge_pipeline/
    └── chunking.py

    scripts/
    └── build_knowledge_chunks.py

    docs/
    └── chunk-schema-v1.md

    tests/
    └── test_chunking_pipeline.py

Chunk 统计只输出到构建命令的控制台并在最终交付报告中汇报，不增加需要同步维护的统计 JSON 文件。

## 4. Chunk Schema v1

`knowledge/chunks.jsonl` 每一行严格符合：

    {
      "schema_version": "1.0",
      "chunk_id": "product:xir-p8668ex:features",
      "parent_document_id": "product:xir-p8668ex",
      "parent_content_hash": "<64-character-lowercase-sha256>",
      "type": "product",
      "section": "产品特点",
      "text": "# 摩托罗拉 XiR P8668Ex\n\n## 产品特点\n\n...",
      "language": "zh-CN",
      "source_url": "https://www.shengborun.com/two-way-radio/xir-p8668ex/",
      "source_files": [
        "src/content/product-categories/two-way-radio.json",
        "src/content/products/two-way-radio/xir-p8668ex.json"
      ],
      "metadata": {
        "product_id": "xir-p8668ex",
        "slug": "xir-p8668ex",
        "category_id": "two-way-radio",
        "category_name": "对讲机通信"
      }
    }

字段职责：

- `schema_version`：Chunk contract 版本，V1 固定为 `1.0`。
- `chunk_id`：稳定、可读、确定的 Chunk 身份，不使用 UUID。
- `parent_document_id`：Day 2 父 Document 身份。
- `parent_content_hash`：父 Document 当前语义内容哈希，用于识别过期 Chunk。
- `type`：`product`、`solution`、`support`、`company`、`contact` 之一。
- `section`：真实、人类可读的原始章节名称，允许中文。
- `text`：未来的 retrieval unit；包含足够父级身份和未经改写的事实内容。
- `language`：继承父 Document，当前为 `zh-CN`。
- `source_url`：继承父 Document 的网页 citation URL。
- `source_files`：继承父 Document 的已排序 provenance 文件列表。
- `metadata`：完整继承 Day 2 已经精简的类型专属 metadata。

不增加 `chunk_hash`。父哈希足以发现父文档变化；整个 Chunk 输出会确定性重建和原子替换。

## 5. Chunk ID Design

Chunk ID 只包含 ASCII，使用冒号分段：

    <parent_document_id>:<semantic-section>

示例：

    product:xir-p8668ex:overview
    product:xir-p8668ex:features
    product:xir-p8668ex:spec:general
    solution:hotel:core-needs
    solution:hotel:body:system-functions

如果一个真实语义章节因长度需要继续拆分，使用确定性数字后缀：

    solution:hotel:body:system-functions:1
    solution:hotel:body:system-functions:2

标准模板章节使用固定 ASCII 名称，例如 `overview`、`features`、`core-needs` 和 `design`。

真实来源标题使用显式维护的稳定 ASCII slug registry。不得自动翻译、自动拼音或静默 fallback。遇到尚未登记的新标题时，构建必须失败，并报告标题和父 Document；维护者显式增加映射后才能重新构建。

`section` 与 `text` 始终保存真实中文标题。中文不会进入程序标识符，因此不影响 Chunk ID 的跨平台稳定性。

## 6. 通用 Chunking Rules

- Chunking 只切分和重复必要身份上下文，不总结、改写、推断或丰富知识。
- 每个 Chunk 必须能脱离父 Document 独立理解。
- 父级名称和必要祖先标题可以作为身份上下文重复。
- 不使用机械百分比 overlap。
- 只有真实语义边界需要时才重复祖先上下文。
- 可选章节不存在时不生成空 Chunk。
- Product 应用场景仅在可信父 Document 原文明示存在时生成；不得从 Solution 或常识推断。
- 不建立 Product–Solution 关系。

## 7. 长度与二次切分

1000 Unicode 字符是 Product、Solution、Support 和 Company 的软上限，不是强制截断线。

真实语义章节超过 1000 字符时：

1. 优先在段落边界切分。
2. 列表只在完整列表项之间切分。
3. 不拆断句子，不截断事实，不加入机械 overlap。
4. 各子 Chunk 重复必要的父级身份与祖先标题。
5. 使用 `:1`、`:2` 等确定性后缀。

如果单个不可再分的完整段落或列表项本身超过 1000 字符，则保持完整并允许 Chunk 超过软上限。

Contact 始终保持一个 Chunk。

## 8. Product Chunking

按父文档原始语义顺序生成：

    overview
    features
    spec:<mapped-group>
    applications  # 仅可信原文存在时

### 8.1 Overview

`overview` 包含产品标题、产品分类和产品介绍。产品身份和 retrieval-critical 名称保留在 Chunk text 中。

### 8.2 Features

`features` 包含产品标题作为身份上下文，以及“产品特点”标题和全部特点原文。

### 8.3 Technical Parameters

现有每个技术参数语义组独立生成一个 Chunk。Chunk text 包含：

- 产品标题。
- “技术参数”祖先标题。
- 参数组真实标题。
- 该组全部参数原文。

例如“`一般规格`”通过显式映射生成：

    product:xir-p8668ex:spec:general

### 8.4 Applications

当前 Day 2 Product 没有可信应用场景，因此不产生 applications Chunk。未来只有父 Document 中出现已确认的真实 applications 章节时才允许生成，不能从 Solution 关系或一般行业知识补充。

## 9. Solution Chunking

标准化后的实际章节分别生成 Chunk，例如：

    summary
    core-needs
    design
    features

其他来源正文按真实 H2 章节切分。采用已确认的整章规则：一个 H2 及其全部 H3、段落和列表内容保持在同一个 Chunk，不为每个 H3 创建子 Chunk。

例如：

    solution:hotel:body:system-functions

若该完整 H2 超过软上限，再按自然边界生成：

    solution:hotel:body:system-functions:1
    solution:hotel:body:system-functions:2

不得创造来源中不存在的 Architecture、Benefits 或其他概念章节。

## 10. Support Chunking

每个 Support Document 默认生成一个 Chunk，并保留服务名称、摘要和内容的真实文本。

只有超过 1000 字符时才按段落或完整列表项边界二次切分。

## 11. Company Chunking

若父 Document 存在真实语义章节，则按真实章节切分；否则保持一个 Chunk。

不得为达到目标长度而人为创造公司章节。超过 1000 字符时应用通用自然边界规则。

## 12. Contact Chunking

Contact 始终生成一个 Chunk，保留联系主题和全部可信联系字段，不因字符数进行拆分。

## 13. Deterministic Ordering and Serialization

输出顺序：

1. 严格遵循 `documents.jsonl` 中父 Document 的既定顺序。
2. 每个父 Document 内遵循原文语义顺序。
3. Product 使用 `overview → features → technical parameter groups → applications`。
4. Solution 的标准章节在前，其他真实 H2 章节按原文顺序。
5. 二次切分按 `:1 → :2 → ...`。

序列化规则：

- 顶层字段顺序固定。
- 每行一个 compact JSON object。
- UTF-8 编码，中文保持可读，不转为 ASCII escape。
- 换行统一为 LF。
- 文件末尾恰好一个换行。
- 不写入时间戳或运行相关随机值。

相同有效输入必须产生逐字节完全相同的 `chunks.jsonl`。

## 14. Validation

### 14.1 Input Validation

Day 3 在切分前重新验证 `documents.jsonl` 的 JSONL 语法和 Day 2 Document contract。错误必须包含：

- 输入文件路径。
- JSONL 行号。
- 能识别时的 `document_id`。
- 具体失败原因。

### 14.2 Chunk Validation

- 所有 Chunk Schema 字段存在，未知字段被拒绝。
- `chunk_id` 全局唯一并符合允许的 ASCII 格式。
- `parent_document_id` 对应本次输入中的真实 Document。
- `parent_content_hash` 与父 Document 当前 `content_hash` 一致。
- `type`、`language`、`source_url`、`source_files` 和 `metadata` 与父 Document 一致。
- `section` 非空，且符合对应类型的固定章节或显式标题映射。
- `text` trim 后非空。
- 每个输入父 Document 至少产生一个 Chunk。
- 输出顺序符合确定性排序规则。

### 14.3 No Silent Knowledge Loss

类型解析器把父 Document 登记为一组真实语义内容块。每个内容块必须：

- 被明确分配给至少一个 Chunk。
- 以原文出现在相应 Chunk 中。
- 不被总结、改写或遗漏。

任何未登记、未分配或内容不匹配都会使整个构建失败。

不要求把所有 Chunk 简单拼接后与父 `text` 逐字节相同，因为 Chunk 会重复父级身份和祖先标题。重复的身份上下文不视为事实重复错误。

## 15. Error Handling and Atomic Output

构建步骤：

1. 读取并验证全部父 Documents。
2. 在内存中解析并生成全部 Chunks。
3. 验证 Chunk schema、集合不变量和内容覆盖。
4. 写入同目录临时文件。
5. 重新读取临时 JSONL 并验证。
6. 全部成功后原子替换 `knowledge/chunks.jsonl`。

任一步失败时以失败状态退出，清理临时文件，并保持已有 `chunks.jsonl` 逐字节不变。不生成可被误认为成功结果的部分输出。

## 16. Statistics

成功构建后输出：

- total documents。
- total chunks。
- average chunk length。
- median chunk length。
- minimum chunk length。
- maximum chunk length。
- chunks per document。
- chunks per type。

所有长度统一以 Unicode 字符数计算并明确标注单位为“字符”。本阶段不引入 tokenizer。

## 17. Testing Strategy

保留并运行全部 Day 2 测试。Day 3 增加聚焦测试：

- Chunk Schema valid/invalid cases。
- Product overview、features 和各参数组切分。
- Product 不生成推断 applications，不建立 Product–Solution 关系。
- Solution 标准章节与真实 H2 切分，H3 保留在所属 H2 Chunk 中。
- Support、Company、Contact 的类型规则。
- 1000 字符软上限、自然边界、不可拆完整块和确定性数字后缀。
- 显式中文标题到 ASCII slug 映射。
- 未映射来源标题构建失败。
- Chunk ID uniqueness、parent validity 和 inherited field consistency。
- 每个父 Document 至少一个 Chunk。
- 语义内容块覆盖和原文保存。
- 父 Document 无效时报告文件、行号、ID 和原因。
- UTF-8 中文、LF、固定字段顺序和逐字节确定性。
- 构建失败时已有输出保持不变。
- 统计计算使用 Unicode 字符数。

Integration test 读取真实 Day 2 `documents.jsonl`，验证全部 60 个当前父 Documents 均有 Chunk，但实现不把 60 写成永久业务常量。

## 18. Acceptance Criteria

- Day 3 只读取已验证的 `knowledge/documents.jsonl`。
- 当前 60 个父 Documents 每个至少生成一个 Chunk。
- 所有 Chunk 符合 Chunk Schema v1。
- 每个 Chunk 保留足够父身份，脱离父文档仍可理解。
- Product、Solution、Support、Company 和 Contact 遵循各自已确认规则。
- 不存在推断 applications 或 Product–Solution 关联。
- 父 Document 的每个真实语义内容块均被原文覆盖。
- 不存在重复 Chunk ID、无效父引用或 provenance 漂移。
- 相同输入重复构建得到逐字节相同输出。
- 失败不覆盖已有 `chunks.jsonl`。
- 控制台输出要求的字符长度和数量统计。
- 全部 Day 2 与 Day 3 自动化测试通过。
- 实现中不存在实验 Chunker、评估框架、embedding、vector store、retrieval、reranking 或 LLM logic。
