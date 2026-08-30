# RAG Knowledge Documents V1 Design

**状态：** 对话设计已确认，等待书面 spec 审阅  
**日期：** 2026-08-30  
**范围边界：** 回答“What knowledge do we have, and how should it be represented?”，在 chunk 之前停止。

## 1. 目标

为 AI Customer Support Assistant 建立 RAG 的第一段 ingestion pipeline，把人工整理的网站知识转换成确定、可验证、可重复生成的 Normalized Documents。

数据流：

    Shengborun Website Data
            ↓ 一次性人工提取与录入
    knowledge/source/**/*.json
            ↓ Extract（读取 JSON）
    Type-specific Source Validation
            ↓ Normalize
    Knowledge Document Schema v1 Validation
            ↓ Atomic Serialize
    knowledge/documents.jsonl

本阶段不包含 chunk、embedding、vector database、similarity search、retrieval 或 LLM。

## 2. 仓库边界与所有权

pipeline 代码、人工维护的数据、文档、测试和生成结果全部位于 AI-Customer-Support-Assistant。

相邻的 Shengborun Astro 仓库只作为人工录入参考与 provenance 来源。构建过程不得读取 D:\Shengborun，不依赖该目录存在，也不在运行时解析 Astro、TypeScript、Markdown 或网站构建产物。

网站知识首次人工录入 knowledge/source/。此后，可重复构建只读取这些 JSON。

## 3. 正式交付物

    knowledge/
    ├── source/
    │   ├── products/
    │   │   └── <product-id>.json
    │   ├── product-categories/
    │   │   └── <category-id>.json
    │   ├── solutions/
    │   │   └── <solution-id>.json
    │   ├── support/
    │   │   └── <service-id>.json
    │   ├── company.json
    │   └── contact.json
    └── documents.jsonl

    scripts/
    └── build_knowledge_documents.py

    docs/
    └── knowledge-schema-v1.md

自动化测试也必须进入项目现有测试结构。

## 4. 当前知识盘点

初始库存：

- 49 个 published Product。
- 4 个 published Product Category，仅用于 Product 验证与上下文。
- 6 个 published Solution。
- 3 个 published Support Service。
- 1 个 Company。
- 1 个 Contact。

首次输出预计为 60 份文档：

    product: 49
    solution: 6
    support: 3
    company: 1
    contact: 1

60 是当前库存结果，不是永久硬编码的验证规则。

## 5. Inclusion / Exclusion

### 5.1 Include

- Product 名称、分类身份、特点、介绍和技术参数。
- Solution 名称、摘要、核心需求、方案设计、特点和完整文字正文。
- 每项 Support Service 的名称、摘要和正文。
- About 页“公司简介”下的四段事实正文。
- Contact 中的公司名称、值班电话和电子邮箱。

### 5.2 Exclude

- Navigation、breadcrumb、footer、copyright 和其他页面结构。
- Homepage hero、营销卖点、选择理由、卡片文案和 CTA。
- Homepage 中与 Product Category、Solution、Support 重复的摘要。
- SEO title、description、path、image。
- 图片、图片路径、尺寸、位置、gallery、diagram 和 alt。
- icon、sort order、视觉配置和其他展示字段。
- About hero、Contact CTA、Support hero 和 Support CTA。
- 公司地址。
- Privacy 页面重复的联系方式。
- Product 应用场景、推荐用途和推断出的 Product–Solution 关系。
- Product 到 Solution 或 Solution 到 Product 的任何关系字段。

V1 不包含“产品选型转人工”的知识规则。唯一约束是 Product 文档不得包含推断出的应用推荐或 Solution 关联。

### 5.3 Product Category 边界

所有 Category 都保留为 Source JSON，用于验证 Product 的 category_id，并把 Category 名称写入 Product text 与 metadata。

当前四个 Category 只有名称和一句 short description，不生成独立 Normalized Document。未来只有当 Category 拥有独立、用户可问的详细知识后，才在新 schema 版本中考虑 Category Document。

## 6. Source JSON Design

### 6.1 公共规则

- 每个文件只有一个顶层 JSON object，不使用顶层 array。
- 字段统一使用 snake_case。
- Schema 为 strict，未知字段直接报错。
- 必填字符串 trim 后必须非空。
- 必填数组必须存在并至少有一项。
- null 不能代替缺失的 required value。
- 每个实体必须显式填写 published。
- published 为 false 的记录仍需通过 source schema，但不生成文档。
- ID 与 slug 使用小写 kebab-case。
- 实体目录中的文件名必须等于 <id>.json。
- 不改写事实语言、标点、数字、型号或单位。

### 6.2 Provenance

每个 Source Record 包含非空 provenance：

    "provenance": [
      {
        "kind": "website_file",
        "reference": "src/content/products/two-way-radio/xir-p8668ex.json"
      }
    ]

V1 中 kind 只允许 website_file。reference 相对于 Shengborun 仓库根目录。禁止绝对路径、盘符、父目录穿越和 URL。

由于构建独立于网站仓库，它只验证 provenance 格式，不能证明所引用的网站文件当前真实存在。

### 6.3 Product Source Schema

    id
    name
    slug
    category_id
    key_features[]
    product_features
    technical_parameters[]
    source_path
    published
    provenance[]

技术参数组：

    {
      "group": "一般规格",
      "items": [
        {
          "name": "频率范围",
          "value": "UHF：403-470MHz VHF：136-174MHz"
        }
      ]
    }

key_features、technical_parameters、每个参数组的 items 和其中所有字符串均 required 且 non-empty。

排除图片、gallery、sort order、SEO、应用场景和 Solution reference。

### 6.4 Product Category Source Schema

    id
    name
    slug
    short_description
    source_path
    published
    provenance[]

所有字段 required 且 non-empty。V1 中 Category 只作 lookup 与 validation。

### 6.5 Solution Source Schema

    id
    name
    slug
    summary
    core_needs[]
    solution_design
    features[]
    body_markdown
    source_path
    published
    provenance[]

稳定 id 初始等于现有网站 slug，但两者保持独立。未来路由变化可以修改 slug，而 document identity 不变。

body_markdown 保留标题、段落、有序列表和无序列表，不在 Normalize 阶段切分。排除图片、sort、SEO 和 Product 关联。

### 6.6 Support Service Source Schema

    id
    name
    summary
    body
    source_path
    published
    provenance[]

每个 Service 生成一份文档。排除页面 Hero、CTA、icon 和重复展示文字。

### 6.7 Company Source Schema

    id
    name
    introduction[]
    source_path
    published
    provenance[]

初始 id 为 shengborun。introduction 按原顺序保存 About 页面四个非空段落。

### 6.8 Contact Source Schema

    id
    company_name
    duty_phone
    email
    source_path
    published
    provenance[]

初始 id 为 shengborun。排除地址、CTA、tel: 与 mailto: 派生链接，以及产品选型指令。

## 7. Knowledge Document Schema v1

documents.jsonl 每一行都严格符合：

    {
      "schema_version": "1.0",
      "document_id": "product:xir-p8668ex",
      "type": "product",
      "title": "摩托罗拉 XiR P8668Ex",
      "text": "# 摩托罗拉 XiR P8668Ex\n\n...",
      "language": "zh-CN",
      "source_path": "/two-way-radio/xir-p8668ex/",
      "source_url": "https://www.shengborun.com/two-way-radio/xir-p8668ex/",
      "source_files": [
        "src/content/product-categories/two-way-radio.json",
        "src/content/products/two-way-radio/xir-p8668ex.json"
      ],
      "content_hash": "<64-character-lowercase-sha256>",
      "metadata": {
        "product_id": "xir-p8668ex",
        "slug": "xir-p8668ex",
        "category_id": "two-way-radio",
        "category_name": "对讲机通信"
      }
    }

字段职责：

- schema_version：契约版本，固定为 1.0。
- document_id：稳定全局身份，格式为 <type>:<entity-id>。
- type：product、solution、support、company、contact 之一。
- title：人类可读标题。
- text：未来 embedding input 和 LLM context。
- language：固定为 zh-CN，不写入 text。
- source_path：网站相对路径，可带 fragment。
- source_url：base URL 加 source_path 得到的绝对 citation URL。
- source_files：排序、去重后的网站仓库相对 provenance 路径。
- content_hash：语义内容的确定性 SHA-256。
- metadata：小型、类型专属的 filtering、debugging 和 identity payload。

默认 base URL 为 https://www.shengborun.com，并允许显式配置覆盖。Source JSON 不重复人工填写绝对 URL。

### 7.1 Type-specific Metadata

    product:
      product_id
      slug
      category_id
      category_name

    solution:
      solution_id
      slug

    support:
      service_id

    company:
      company_id

    contact:
      contact_id

大段业务知识不能只存在 metadata，也不在 metadata 中重复。

### 7.2 Content Hash

Hash input 是以下字段的 canonical deterministic serialization：

    type
    title
    text
    language
    metadata

算法为 SHA-256，输出 64 位小写十六进制。

以下字段不参与 hash：

    schema_version
    document_id
    source_path
    source_url
    source_files
    timestamps
    absolute filesystem paths

## 8. Normalization Rules

Normalization 只重组已知事实，不改写事实。

允许：

- 换行统一为 LF。
- 删除行尾空白。
- trim 字段最外层多余空白。
- 按确定模板放置标题、段落和列表。
- 插入模板所需固定标签。

禁止润色正文、修正或统一标点、改变技术单位、推断缺失事实、增加应用场景，或建立 Product–Solution 关系。

### 8.1 Product Text

    # {name}

    产品分类：{category_name}

    ## 产品介绍

    {product_features}

    ## 产品特点

    - {key_feature}

    ## 技术参数

    ### {group}

    - {parameter_name}：{parameter_value}

Product metadata：

    product_id
    slug
    category_id
    category_name

Product source_files 合并 Product provenance 与所引用 Category provenance。

### 8.2 Solution Text

    # {name}

    ## 方案摘要

    {summary}

    ## 核心需求

    - {core_need}

    ## 方案设计

    {solution_design}

    ## 方案特点

    - {feature}

    ## 详细内容

    {body_markdown}

Metadata 为 solution_id 和 slug。

### 8.3 Support Text

    # {name}

    ## 服务摘要

    {summary}

    ## 服务内容

    {body}

Metadata 为 service_id。

### 8.4 Company Text

    # {name}

    ## 公司简介

    {introduction paragraph 1}

    {introduction paragraph 2}

全部段落按原顺序排列，每段之间一个空行。Metadata 为 company_id。

### 8.5 Contact Text

    # 联系我们

    公司名称：{company_name}
    值班电话：{duty_phone}
    电子邮箱：{email}

Metadata 为 contact_id。

## 9. Validation

### 9.1 Source Validation

- JSON syntax error 必须包含 file、line 和 column。
- 应用 strict type-specific schema。
- 验证 filename 与 ID 一致。
- 验证 kebab-case ID 和 slug。
- 验证 email、telephone、source_path 和 provenance 格式。
- source_path 必须以 / 开头，禁止域名、query string 和 path traversal。
- 每个记录都验证，包括 unpublished records。

### 9.2 Cross-source Validation

- 同一 source type 内实体 ID 唯一。
- 每个 published Product 引用存在且 published 的 Category。
- Product path 与 Category slug、Product slug 一致。
- Solution、Support、Company、Contact path 符合各自路由规则。
- Product 与 Solution 无法携带互相关联字段，因为未知字段会被拒绝。

### 9.3 Document Validation

- 所有顶层字段存在，未知字段被拒绝。
- type、schema_version 和 language 只能使用允许值。
- document_id 与 source type、source ID 一致。
- title 和 text 非空。
- source_url 严格等于 configured base URL 加 source_path。
- 重新计算的 content_hash 必须一致。
- document_id 全局唯一。
- 不同 document_id 出现相同 content_hash 时直接失败。
- Product、Solution、Support、Company、Contact 每类至少一份。
- 报告各类型数量，但不把总数永久固定为 60。

## 10. URL Validation

默认构建离线运行。强制的 path 与 route validation 不访问网络。

显式 --check-urls 模式先完成所有离线验证，再检查每个唯一的绝对页面 URL。发送 HTTP 请求前移除 #company、#contact 等 fragment。允许跟随 redirect，最终 response 必须成功。

在线验证需区分页面不存在与网络或 transport failure。可选在线验证失败时不得替换现有 documents.jsonl。

## 11. Error Handling and Atomic Output

一次运行应尽量汇总所有能安全发现的错误。Diagnostic 格式：

    [ERROR] knowledge/source/products/xir-p8668ex.json
    entity: product:xir-p8668ex
    field: technical_parameters[0].items[2].value
    reason: required string must not be empty

跨文件错误同时指出引用记录与缺失或无效的被引用实体。

写入顺序：

1. 读取所有 Source Records。
2. 验证 Source 与 cross-source relationships。
3. 在内存中 Normalize 所有 published entities。
4. 验证所有 Documents 和 collection-wide invariants。
5. 在输出目录写 temporary file。
6. 重新读取并逐行验证 temporary JSONL。
7. 全部成功后才原子替换 knowledge/documents.jsonl。

失败时删除临时文件，以失败状态退出，并保持旧输出逐字节不变。

## 12. Deterministic Serialization

- 按 document_id 字典序排序。
- source_files 排序并去重。
- 顶层字段顺序固定。
- 每行一个 compact JSON object。
- UTF-8 编码，中文保持可读，不转为 ASCII escape。
- 所有平台统一 LF。
- 文件末尾恰好一个换行。
- 不包含 created_at、updated_at、generated_at 或任何 timestamp。

相同的 validated inputs 和 configuration 必须产生逐字节相同的结果。

## 13. Testing Strategy

自动化测试覆盖：

- 每种 Source Schema 的 valid 和 invalid cases。
- Unknown field、empty value、empty array、invalid ID、path 和 provenance。
- Product 引用 missing 或 unpublished Category。
- 五个 normalizer 的 text、metadata、source files、ID 和 hash。
- Product–Solution relation 与 Product application recommendation 不存在。
- Hash determinism 和 inclusion/exclusion contract。
- Duplicate document ID 与 duplicate content hash failure。
- UTF-8 Chinese、LF、fixed order、one-object-per-line JSONL。
- 失败后现有输出保持不变。
- Base URL joining 和 online check fragment removal。

Integration test 根据 published source entities 动态计算 expected count，不永久写死 60。

人工抽查至少覆盖：

- 一个包含多个技术参数组的 Product。
- 一个标题层级较多的长 Solution。
- 一个 Support Service。
- Company。
- Contact。

人工检查字段完整、参数原值、Markdown 可读性、text 中的身份、citation 与 provenance、噪声排除、Product–Solution 关系缺失和重复知识缺失。

## 14. Documentation Requirements

docs/knowledge-schema-v1.md 必须解释：

- What enters and does not enter the Knowledge Base。
- 每类 Source Record 的 schema。
- Normalized Document 的定义。
- Text 与 metadata 的职责。
- Document IDs、source path、citation URL、provenance 和 hash。
- 每类 transformation。
- Validation 与 failure behavior。
- Build usage 和 optional URL checking。
- 明确在 chunk 前停止。

## 15. Acceptance Criteria

- 每个 curated Source JSON 通过 strict schema。
- 初始库存生成 49 Product、6 Solution、3 Support、1 Company、1 Contact。
- 合法新增或删除 published record 后，构建无需修改永久数量常量。
- 每一行通过 Knowledge Document Schema v1。
- 不存在 duplicate ID 或 duplicate semantic hash。
- 相同输入重复构建得到逐字节相同输出。
- 所有自动化测试通过。
- 人工抽查无字段丢失、事实改写、错误拼接、重复、噪声或虚假的 Product–Solution 关系。
- 实现不包含 chunk、embedding、vector database、retrieval 或 LLM logic。

