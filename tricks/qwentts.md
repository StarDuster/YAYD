# Qwen TTS 防劣化与优化机制说明

当前项目中针对 Qwen3-TTS 实施了多层面的防劣化（Anti-Deterioration）和 Prompt 优化机制，旨在解决自回归模型常见的“长语音生成死循环”、“无法停止（Failure-to-Stop）”、“幻觉复读”以及“代码朗读异常”问题。

## 1. 强制 Token 上限限制 (`max_new_tokens`)

为了防止模型无限生成，在底层的 Worker 脚本 (`scripts/qwen_tts_worker.py`) 中设置了硬性的 Token 生成上限。

*   **默认值**: 2048 tokens
*   **配置**: 可通过环境变量 `YOUDUB_QWEN_TTS_MAX_NEW_TOKENS` 修改。
*   **作用**: 当生成长度达到此限制时，强制停止生成。这是最底层的防线。

## 2. 劣化检测：命中上限判定 (Hit Cap Detection)

仅仅限制长度是不够的，如果模型生成到了上限，通常意味着它没有正常生成结束符（EOS），输出的内容往往是重复的、无意义的或者截断的。代码中通过检查生成音频的时长来判断是否发生了这种情况。

*   **原理**: Qwen3-TTS 的音频 Token 采样率约为 12Hz (即每秒约 12 个 tokens)。
*   **计算公式**: 理论最大时长 `MaxDuration = max_new_tokens / 12.0`。
*   **判定逻辑**:
    *   如果 `生成的音频时长 >= MaxDuration - 容差值`，则判定为“劣化”（Degenerate）。
    *   容差值 (`YOUDUB_QWEN_TTS_HIT_CAP_TOL_SEC`) 默认为 2.0 秒。
*   **代码位置**: `src/youdub/steps/synthesize_speech.py` 中的 `_qwen_tts_is_degenerate_hit_cap` 函数。

## 3. 自动重试与 Worker 重启

一旦检测到劣化（即命中 Token 上限），系统会采取以下措施：

1.  **标记失败**: 该片段被标记为生成失败，错误信息为 `qwen_tts_degenerate_hit_max_new_tokens`。
2.  **丢弃结果**: 删除生成的异常音频文件。
3.  **自动重试**: 触发重试逻辑，重新生成该片段。
4.  **Worker 重启**: 为了防止模型陷入某种糟糕的内部状态（虽然通常是无状态的，但为了稳健性），或者为了应对显存碎片等问题，连续失败可能会触发 Worker 进程的重启 (`_restart_qwen_worker`)。

## 4. 通用时长卫士 (Duration Guard)

除了针对 Qwen 的特定检测，还有一个通用的时长检查机制，防止生成的语音过长（通常也是劣化的一种表现，如大量静音或复读）。

*   **检查**: 生成的语音时长不能超过 `(原参考时长 * 比例 + 额外缓冲)`。
*   **作用**: 如果 Qwen 生成了超长语音但未达到 Token 上限（例如中间包含大量静音），此机制会拦截并触发重试。

## 5. Prompt 文本预处理与 Trick

为了让 TTS 更好地朗读技术类内容（如代码库介绍），并减少因特殊符号引起的生成异常，我们在将文本送入模型前进行了一系列预处理（Normalization）。

### 5.1 代码符号发音优化 (`_tts_prompt_normalize_python_tokens`)

针对 Python 风格的代码标识符做了专门的转换，使其发音更自然，避免模型因为无法理解符号而产生幻觉或停顿。

*   **点号转义 (Dot Expansion)**:
    *   **问题**: TTS 容易忽略 `matplotlib.pyplot` 中的 `.`，或者将其读作句号停顿。
    *   **处理**: 将 `object.method` 形式的标识符转换为 `object dot method`。
    *   **示例**: `matplotlib.pyplot` -> `matplotlib dot pyplot`。
    *   **注意**: 仅针对标识符结构，不影响小数（如 `3.14` 保持不变）。

*   **下划线展开 (Underscore Expansion)**:
    *   **问题**: `base_dir` 可能被读成连在一起的词，或者忽略下划线。
    *   **处理**: 将 `snake_case` 替换为空格。
    *   **示例**: `base_dir` -> `base dir`。

### 5.2 递进式文本简化 (Progressive Text Simplification)

当 TTS 生成失败触发重试时，我们会逐步简化输入的 Prompt 文本，以降低生成难度 (`_tts_text_for_attempt`)。

*   **Attempt 0 (首次尝试)**:
    *   仅应用上述的“代码符号发音优化”。
    *   保留大部分原始格式。
*   **Attempt 1 (重试 1)**:
    *   **去除 Markdown 代码标记**: 移除 ` ``` ` 块标记和行内 `` ` ``。
    *   保留文本内容，但去掉格式干扰。
*   **Attempt 2+ (重试 2 及以上)**:
    *   **激进清洗**: 仅保留中文字符、ASCII 字母数字、空格以及基本标点（`，。！？,.!?:：；;（）()\-\+*/=`）。
    *   移除所有其他可能导致模型困惑的特殊符号。

## 总结

完整的防劣化与优化流程如下：

1.  **预处理**: 应用 Dot/Underscore 转义，根据重试次数简化文本。
2.  **生成**: 设置 `max_new_tokens` 进行生成。
3.  **后处理检查**:
    *   检查是否 **Hit Cap** (时长接近 Token 上限对应的时长) -> 视为劣化。
    *   检查是否 **过长** (超过预期的合理时长范围) -> 视为异常。
4.  **应对**: 丢弃 -> 重启 Worker (可选) -> 递进式简化文本 -> 重试。
