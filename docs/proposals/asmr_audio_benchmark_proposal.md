# ASMR 音频生成评测方案（草案）

> 状态：思路草案，尚未实现
> 归属能力维度：内容生成层 → 文生音频
> 关联 Issue：#41

---

## 1. 背景与目标

ASMR（Autonomous Sensory Meridian Response，自发性知觉经络反应）音频是一种通过特定声音触发听众愉悦感和放松反应的音频类型。常见触发音包括耳语、轻敲、刷子摩擦、雨声、翻书声等。

随着 AI 音频生成技术的快速发展，多个大模型平台已具备从文本描述生成音频的能力。本方案旨在评测各 AI 模型/平台生成 ASMR 音频的质量，帮助超级个体（自媒体创作者、助眠内容制作者等）选择最适合的工具。

---

## 2. 可选模型与平台调研

以下是当前（截至 2026 年 2 月）提供音频生成 API 的主要大模型及平台，按类别整理：

### 2.1 综合大模型平台（含 TTS / 音频生成能力）

| 平台/模型 | 音频能力 | API 可用性 | 与 ASMR 的相关性 |
|-----------|---------|-----------|-----------------|
| **OpenAI gpt-4o-mini-tts** | 高可控性 TTS，支持语气/情绪/语速指令 | ✅ 公开 API | 可通过 prompt 控制耳语、轻柔语调等 ASMR 人声 |
| **Google Gemini 2.5 (Flash/Pro TTS)** | 原生音频输出，支持 24+ 语言，可控语气/语调/语速 | ✅ Gemini API | 自然语言控制语音风格，适合 ASMR 人声场景 |
| **Google Lyria 3** | 音乐生成模型，支持风格/人声/节奏控制 | ✅ Gemini App 内 | 可生成 ASMR 背景音乐/氛围音 |
| **ByteDance 豆包 (Doubao)** | 实时语音大模型，高自然度和情感表达 | ✅ 火山引擎 API | 情感连续性强，适合 ASMR 旁白/耳语 |

### 2.2 专业音频/音乐生成平台

| 平台/模型 | 音频能力 | API 可用性 | 与 ASMR 的相关性 |
|-----------|---------|-----------|-----------------|
| **ElevenLabs (Sound Effects V2)** | TTS + 音效生成，48kHz 专业音质，支持 30s 片段和无缝循环 | ✅ 公开 API | 音效生成能力强，适合 ASMR 环境音/触发音（敲击、刷子等） |
| **ElevenLabs (TTS)** | 5000+ 声音，70+ 语言，高度可控 | ✅ 公开 API | 可定制耳语/轻柔人声 |
| **Suno V5** | 全曲生成（含人声+器乐），ELO 1293 | ⚠️ 无官方公开 API（第三方可用） | 可生成 ASMR 风格环境音乐 |
| **Udio V1.5** | 音乐生成，支持人声和歌词 | ⚠️ 无官方公开 API（第三方可用） | 可生成 ASMR 氛围音乐 |
| **MiniMax Music 2.0 (海螺音乐)** | 完整歌曲生成（含人声），最长 5 分钟 | ✅ 公开 API，$0.035/次 | 可生成 ASMR 轻音乐/氛围音 |
| **Stability AI Stable Audio 2.5** | 音乐和音效生成 | ✅ API 可用 | 环境音效、白噪声类 ASMR 内容 |

### 2.3 开源模型（可本地部署）

| 模型 | 音频能力 | 部署方式 | 与 ASMR 的相关性 |
|------|---------|---------|-----------------|
| **Meta AudioCraft (MusicGen + AudioGen)** | 音乐生成 + 音效生成，32kHz | Hugging Face / 本地 | 环境音/音效类 ASMR |
| **Stability AI Stable Audio Open** | 短音频和音效生成 | Hugging Face / 本地 | 触发音效类 ASMR |
| **Fish Speech (OpenAudio S1)** | 高质量 TTS，支持 8 语言，支持语音克隆 | GitHub 开源 / API | 定制 ASMR 人声角色 |
| **CosyVoice 2 (阿里)** | 多语言 TTS | 开源 | ASMR 人声生成 |
| **IndexTTS-2** | 高级 TTS | 开源 | ASMR 人声生成 |

### 2.4 推荐优先评测的模型

根据 API 可用性、音频质量和 ASMR 场景适配度，建议优先评测以下模型：

1. **ElevenLabs Sound Effects V2** — 音效类 ASMR 的最佳候选
2. **ElevenLabs TTS** — 人声类 ASMR（耳语、轻声细语）
3. **OpenAI gpt-4o-mini-tts** — 可控性强的人声 ASMR
4. **Google Gemini 2.5 TTS** — 多语言人声 ASMR
5. **MiniMax Music 2.0** — 背景音乐/氛围类 ASMR
6. **Stable Audio 2.5** — 环境音效类 ASMR
7. **Meta AudioCraft** — 开源基线对照

---

## 3. ASMR 音频评测维度设计

ASMR 音频不同于一般音乐或语音，其评测需要特殊维度。以下是建议的评测框架：

### 3.1 ASMR 音频分类

评测应覆盖 ASMR 的主要类型：

| 类型 | 描述 | 示例 prompt |
|------|------|-----------|
| **人声耳语型** | 轻柔耳语、低声细语 | "Generate a soft whisper voice saying goodnight, with gentle breathing sounds" |
| **触发音效型** | 敲击、刮擦、翻页等物理触发音 | "Generate the sound of fingernails gently tapping on a wooden surface, slow and rhythmic" |
| **环境氛围型** | 雨声、篝火、森林等自然环境音 | "Generate a gentle rain falling on leaves with distant thunder" |
| **角色扮演型** | 模拟场景的人声+音效混合 | "Generate a librarian whispering while slowly turning pages of an old book" |
| **音乐氛围型** | 极低 BPM 的轻柔背景音乐 | "Generate ultra-calming ambient music at 60 BPM with soft piano and subtle wind sounds" |

### 3.2 评测维度

#### A. 客观技术指标（可自动化）

| 指标 | 说明 | 评测方法 |
|------|------|---------|
| **音频质量 (Audio Quality)** | 采样率、比特率、信噪比 | 检测输出文件元数据，计算 SNR |
| **频谱特征** | ASMR 音频通常集中在中低频，高频柔和 | FFT 频谱分析，检查频率分布是否符合 ASMR 特征 |
| **响度一致性** | ASMR 要求整体音量低且稳定，避免突然的音量跳变 | LUFS 响度分析，检测峰值变化 |
| **时长准确性** | 生成的音频时长是否符合要求 | 直接对比 |
| **循环性 (Loopability)** | 音频首尾是否能无缝衔接（环境音场景重要） | 首尾交叉相关性检测 |

#### B. 声学感知指标（半自动化）

| 指标 | 说明 | 评测方法 |
|------|------|---------|
| **PESQ / POLQA** | 感知语音质量评估 | 标准算法，适用于人声类 ASMR |
| **响度 (Loudness)** | 研究表明 ASMR 与低响度正相关 | 计算长期平均响度 |
| **粗糙度 (Roughness)** | 自然声源的 ASMR 与适度粗糙度相关 | 心理声学模型计算 |
| **IACC (双耳互相关系数)** | 人声 ASMR 与低 IACC 相关（更强的空间感） | 双声道互相关分析 |

#### C. 内容匹配度（需 LLM 辅助判断）

| 指标 | 说明 | 评测方法 |
|------|------|---------|
| **Prompt 忠实度** | 生成的音频是否符合 prompt 描述 | 使用音频理解模型（如 Gemini）转译后与原 prompt 对比 |
| **声音元素完整性** | prompt 中提及的声音元素是否都出现了 | 音频分析 + LLM 判断 |
| **时序准确性** | 如果 prompt 描述了时间顺序，音频是否按顺序呈现 | 分段分析 |

#### D. ASMR 效果主观评估（需人工 / LLM-as-Judge）

| 指标 | 说明 | 评分范围 |
|------|------|---------|
| **放松感 (Relaxation)** | 音频是否令人感到放松 | 1-10 |
| **沉浸感 (Immersion)** | 音频的空间感和沉浸程度 | 1-10 |
| **触发强度 (Tingle Factor)** | 是否能有效触发 ASMR 感受 | 1-10 |
| **自然度 (Naturalness)** | 声音是否自然，无明显人工痕迹 | 1-10 |
| **舒适度 (Comfort)** | 是否有令人不适的杂音、爆音或频率 | 1-10 |

---

## 4. 评测流程设计思路

### 4.1 整体流程

```
定义 ASMR 任务集 → 调用各模型 API 生成音频 → 客观指标自动分析 → 音频转文字描述 → LLM-as-Judge 评分 → 人工抽样验证 → 汇总报告
```

### 4.2 详细步骤

**Step 1：任务定义**
- 编写 15-20 个 ASMR 音频生成 prompt，覆盖上述 5 种 ASMR 类型
- 每个 prompt 附带明确的评测 criteria（类似现有 text_generation.json 格式）
- 定义期望的音频参数（时长、采样率等）

**Step 2：音频生成**
- 对每个 prompt，分别调用各模型 API 生成音频
- 统一输出格式为 WAV/MP3，记录生成耗时和 API 成本
- 每个模型每个 prompt 生成 2-3 次取最佳（控制随机性影响）

**Step 3：客观指标自动评测**
- 用 Python 音频分析库（librosa, scipy）提取频谱、响度、SNR 等指标
- 自动检测是否存在爆音、截断、静音段等缺陷
- 生成客观指标评分表

**Step 4：内容匹配度评测（LLM 辅助）**
- 使用 Gemini 等多模态模型"听"音频并生成文字描述
- 将音频描述与原始 prompt 进行对比，由 LLM-as-Judge 打分
- 这一步类似现有框架的 LLM-as-Judge 评分模式

**Step 5：主观 ASMR 效果评测**
- P1 阶段：人工盲听评分（每个音频由 3+ 评审者独立打分）
- P2 阶段：探索用音频理解大模型自动打分
- 评分维度包含放松感、沉浸感、触发强度、自然度、舒适度

**Step 6：汇总报告**
- 按模型、按 ASMR 类型汇总得分
- 输出综合排行榜和各维度对比表
- 附带最佳/最差案例的音频样本链接

### 4.3 与现有框架的集成思路

```
benchmarks/
  text_generation.json     # 已有
  asmr_audio.json          # 新增：ASMR 音频评测任务集

agentbench/
  models/
    audio/                 # 新增：音频模型适配层
      base.py              # AudioModel 抽象基类 (generate_audio(prompt) -> AudioFile)
      elevenlabs_model.py
      openai_tts_model.py
      gemini_tts_model.py
      minimax_music_model.py
      stable_audio_model.py
  evaluators/
    audio_evaluator.py     # 新增：音频客观指标评测器
    audio_llm_judge.py     # 新增：音频内容匹配度 LLM 评分器
```

关键设计点：
- 音频模型的基类需要扩展现有 `BaseModel`，输出从 `str` 变为音频文件路径
- 评测器需要新增音频分析能力，不能复用纯文本的 LLM-as-Judge
- 内容匹配度评测需要引入多模态 LLM（如 Gemini）作为"音频翻译"中间层

---

## 5. 关键挑战与风险

| 挑战 | 说明 | 应对思路 |
|------|------|---------|
| **API 可用性差异** | Suno/Udio 无官方 API，需要第三方或模拟 | 优先评测有官方 API 的平台，第三方 API 作为补充 |
| **输出格式不统一** | 各平台输出格式、采样率、时长限制不同 | 统一后处理流程，对齐到相同参数后再评测 |
| **主观评测成本高** | ASMR 效果高度主观，个体差异大 | 先做客观指标 + LLM 辅助评测，人工评测作为验证手段 |
| **ASMR 效果难量化** | "tingle" 等 ASMR 体验无标准量化方法 | 结合声学指标（响度、IACC、粗糙度）与主观打分，建立相关性 |
| **多模态 LLM 音频理解有限** | 当前 LLM 对音频细节理解能力还在发展中 | 以人工评测为金标准，LLM 评测作为辅助和加速手段 |
| **成本控制** | 音频生成 API 调用成本可能较高 | 精简任务集，控制重复生成次数，优先评测性价比高的平台 |

---

## 6. 分阶段实施建议

### Phase 1：最小可行评测（MVP）
- 选择 3 个有公开 API 的模型（如 ElevenLabs SFX、OpenAI TTS、Stable Audio）
- 编写 5 个代表性 ASMR prompt（每种类型 1 个）
- 实现客观指标自动评测（频谱、响度、SNR）
- 人工盲听打分（3-5 位评审）
- 输出简单对比表

### Phase 2：扩展评测
- 扩展到 6-8 个模型
- ASMR prompt 增加到 15-20 个
- 引入多模态 LLM 辅助评测内容匹配度
- 建立 ASMR 评测的标准流程和评分体系

### Phase 3：自动化评测
- 完善音频评测器的自动化流程
- 探索 LLM-as-Judge 在音频场景的可靠性
- 集成到 AgentBench 主框架的 `run.py` 评测流程
- 支持定期自动评测和排行榜更新

---

## 7. 参考资料

- [OpenAI Audio Models API](https://platform.openai.com/docs/guides/text-to-speech)
- [Google Gemini TTS](https://ai.google.dev/gemini-api/docs/speech-generation)
- [ElevenLabs Sound Effects API](https://elevenlabs.io/docs/overview/capabilities/sound-effects)
- [MiniMax Music Generation API](https://platform.minimax.io/docs/guides/music-generation)
- [Stability AI Stable Audio](https://stability.ai/stable-audio)
- [Meta AudioCraft GitHub](https://github.com/facebookresearch/audiocraft)
- [Fish Speech GitHub](https://github.com/fishaudio/fish-speech)
- [Suno AI](https://suno.com/)
- [Udio AI](https://www.udio.com/)
- PMC: Sound Quality Factors Inducing the Autonomous Sensory Meridian Response (https://pmc.ncbi.nlm.nih.gov/articles/PMC9598278/)
- PEAQ: Perceptual Evaluation of Audio Quality (ITU-R BS.1387)
