# 翻译策略文档

YouDub-webui 的翻译模块 (`src/youdub/steps/translate.py`) 旨在提供高质量、上下文连贯且术语一致的字幕翻译。

## 核心策略

目前支持两种主要的翻译策略，可通过环境变量 `TRANSLATION_STRATEGY` 或 UI 设置进行配置。

### 1. `guide_parallel` (默认/推荐)

**并发分块翻译 + 全局术语指南**。

这是目前默认且推荐的策略，平衡了速度与质量（一致性）。

*   **流程**：
    1.  **生成全局指南**：首先阅读完整的视频摘要和转录文本，生成一份全局的术语表（Glossary）和注意事项（Notes）。这确保了整个视频中专有名词翻译的一致性（例如 "Agent" 统一翻译为 "智能体"）。
    2.  **智能分块**：将转录文本切分为多个块（Chunk）。
        *   默认块大小由 `TRANSLATION_CHUNK_SIZE` 控制（默认 8 句）。
        *   **说话人感知**：分块逻辑会确保**不跨越说话人边界**。即同一个 Chunk 内的所有句子都来自同一个说话人。这避免了将不同人的对话混淆在一起翻译。
    3.  **并发执行**：使用线程池并发请求 LLM 翻译这些 Chunks。
        *   并发数由 `TRANSLATION_MAX_CONCURRENCY` 控制（默认 4）。
    4.  **上下文注入**：对于每个 Chunk（或 Chunk 内的首句），如果检测到说话人切换，系统会自动注入**上文（Context）**。
        *   Prompt 会包含：“上文（Speaker A）说：‘...’”
        *   明确指示模型只翻译当前句，利用上文消除歧义（如指代词 resolution）。

### 2. `history`

**串行历史翻译**。

*   **流程**：
    *   逐句串行翻译。
    *   维护一个滑动的对话历史窗口（History Window），将之前的原文和译文作为对话历史发送给 LLM。
    *   **说话人切换处理**：同样具备说话人感知能力。当说话人变化时，会在 User Prompt 中显式插入上文提示，帮助模型理解对话流。

---

## 关键特性

### 1. 说话人感知的上下文 (Speaker-Aware Context)

为了解决多说话人对话时的翻译准确性（例如 A 问 B 答），系统引入了显式的上下文注入机制：

*   **隔离**：翻译请求绝不会将不同说话人的句子合并成一段文本让模型翻译（防止模型“脑补”对话关系导致幻觉）。
*   **关联**：当轮到 B 说话时，Prompt 会引用 A 刚才说的话作为背景信息。
*   **Prompt 示例**：
    > 上文（SPEAKER_00）说：“Do you like apples?”
    > 现在（SPEAKER_01）说：“Yes, I do.”
    > 请只翻译“现在说”的这一句...

### 2. 翻译校验与降级 (Validation & Fallback)

为了防止 LLM 产生幻觉或格式错误，系统包含严格的校验逻辑：

*   **校验规则**：
    *   禁止包含“翻译：”、“译文：”等前缀。
    *   禁止包含解释性文本（如“这句话的意思是...”）。
    *   长度检查：译文过短（内容丢失）或过长（啰嗦）会被拒绝。
    *   格式检查：禁止换行、不配对的引号等。
*   **自动修复**：
    *   去除首尾的引号、Markdown 代码块标记。
    *   去除常见的“翻译是：”前缀。
*   **智能降级**：
    *   如果 Chunk 翻译失败（如格式错误），回退到该 Chunk 内**逐句翻译**。
    *   如果单句翻译因“内容丢失（Too short）”连续失败，会尝试将该句**拆分为更小的子句**（Split Sentences）递归翻译，最后合并。

### 3. 标点修复 (Punctuation Fix)

在翻译之前，系统会尝试修复 Whisper 转录结果中的标点符号（`transcript_punctuated.json`）。
*   利用 LLM 修复 ASR 产生的长句（Run-on sentences）、缺失的问号/句号等。
*   更好的断句有助于提高翻译质量。

---

## 配置参数

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `TRANSLATION_STRATEGY` | 策略选择 (`guide_parallel` / `history`) | `guide_parallel` |
| `TRANSLATION_MAX_CONCURRENCY` | 翻译最大并发数 | `4` |
| `TRANSLATION_CHUNK_SIZE` | 单次请求包含的句子数量（仅 guide_parallel） | `8` |
| `TRANSLATION_GUIDE_MAX_CHARS` | 生成指南时参考的原文最大字符数 | `2500` |
| `YOUDUB_PUNCTUATION_FIX_BEFORE_TRANSLATE` | 是否启用翻译前标点修复 | `True` |
