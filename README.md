
## Project Structure (Updated)

This repo now uses a standard `src/` package layout while keeping old commands compatible.

### Install (required)

From repo root:

- Local: `pip install -e .`
- Quick package check: `python -c "import numen_scriptorium; print(numen_scriptorium.__version__)"`

Colab (after `git clone`):

1. `cd` into repo root
2. `pip install -e .`

### New directories

```text
src/numen_scriptorium/
  cli/            # train / infer / smoke entrypoints
  training/       # QLoRA training implementation
  inference/      # adapter loading + generation
  data/           # data module namespace
  common.py
  config.py
  paths.py

configs/
  train_qwen_1_5b.yaml
  train_qwen_7b.yaml

scripts/
  train.py
  infer.py
  smoke.py
  build_dataset.py
  boh_extractor.py
  cs_extractor.py
```

### New recommended commands

Local (from repo root):

- Build dataset:
  `python scripts/build_dataset.py`
- Split JSONL:
  `python scripts/split_jsonl.py --in_file data/train_all.jsonl --val_ratio 0.05 --test_ratio 0.0 --seed 42`

- Train:
  - 7B:
    `python -m numen_scriptorium.cli.train --config configs/train_qwen_7b.yaml`
  - 1.5B:
    `python -m numen_scriptorium.cli.train --config configs/train_qwen_1_5b.yaml`
- Train with overrides (example):
  `python -m numen_scriptorium.cli.train --config configs/train_qwen_7b.yaml --preset t4 --output_dir outputs/qwen2_5_7b_boh_qlora_t4 --resume latest`
- Infer (base + adapter):
  `python -m numen_scriptorium.cli.infer --base_model Qwen/Qwen2.5-7B-Instruct --adapter outputs/qwen2_5_7b_boh_qlora/best`
- Smoke test (no training):
  `python -m numen_scriptorium.cli.smoke --config configs/train_qwen_7b.yaml --max_seq_len 512`

- (Short form)
  `python -m numen_scriptorium.cli.train --config configs/train_qwen_7b.yaml`

Script shortcuts:

- `python scripts/train.py --config configs/train_qwen_7b.yaml`
- `python scripts/infer.py --adapter outputs/.../best`

### Backward compatibility

Old entry files still work:

- `python train_qlora_qwen.py ...`
- `python infer_qlora_qwen3_boh.py ...`
- `python boh_extractor.py`
- `python cs_extractor.py`
- `python build_train_data_from_raw.py`
- `python transfer_raw.py`

They now delegate to the new package CLI.

### Colab note

If running in Colab, `cd` into repo root before running commands above so relative config/data paths resolve correctly.

### Hugging Face download notes (Windows)

- Optional token for higher limits/faster downloads:
  - Windows CMD/PowerShell: `set HF_TOKEN=your_token` (CMD) / `$env:HF_TOKEN="your_token"` (PowerShell)
  - Bash: `export HF_TOKEN=your_token`
- Symlink warning on Windows is expected if Developer Mode is off.
  - You can enable **Windows Developer Mode** or run terminal as Administrator.
  - To silence warning only: set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.

---

# 🇬🇧 English README

# Numen Scriptorium 

Numen Scriptorium is a bilingual text engine based on **Qwen3-0.6B**, fine-tuned to write and translate in a style inspired by occult libraries and intricate game worlds.

The project draws inspiration and training data from games **Book of Hours** and **Cultist Simulator**. By extracting and restructuring their text assets, it trains a model that can:

- Translate between English and Chinese while preserving tone and lore
- Generate in-universe item descriptions and environmental text
- Write worldbuilding lore in a consistent style
- Produce bilingual text suitable for MODs, localization prototypes, and fan works

> ⚠️ **Project Status**  
> This project is currently under active development.  
> At the moment, the fine-tuning dataset only includes **book-related texts** (tomes/journals, etc.) from *Book of Hours*.  
> Support for more types (skills, incidents, visitors, weather, etc.) and for *Cultist Simulator* data will be added gradually.
> ⚠️ This project is for technical research and personal creative experimentation only.  
> It does **not** include or distribute any original game assets.  
> You are responsible for ensuring that your use of game data complies with the relevant games’ EULA and copyright terms.

---

## ✨ Features Overview

### 🧠 Model & Fine-tuning

- Base model: **Qwen3-0.6B**
- Fine-tuning method: **QLoRA (4-bit quantization + LoRA)**  
  - Designed to run on consumer GPUs 
- Focused on:
  - Bidirectional English↔Chinese translation with style preservation
  - Generating full descriptions and stories from structured signals (name, type, aspects, mystery, etc.)
  - Stable bilingual (CN→EN) paired outputs

### 📚 Data Sources & Extraction

- **Book of Hours**
  - Extracts: tomes, journals, aspected items, skills, incidents, abilities, visitors, weather, etc.
  - Parses `xexts` / `xtriggers` to recover reading text and “learning outcomes”

- **Cultist Simulator**
  - Extracts: books, tools, ingredients, fragments, influences, vaults, etc.
  - Uses recipes to derive multi-stage descriptions for vaults (setup/success/etc.)

All data is first stored as **grouped raw JSON**, then converted into unified `instruction / input / output` samples for supervised fine-tuning.

### 🌐 Translation & Style Imitation

- English → Chinese:
  - Keeps core lore and entities intact
  - Mimics BOH/CS’s tone: metaphor, religious myth, ritualistic voice
- Chinese → English:
  - Produces English with a Lovecraftian / occult / gothic flavour
- Supports “Chinese first, then English” bilingual outputs for:
  - MOD text
  - Lore entries
  - In-universe encyclopedias

### 🧩 Structured → Text Generation

- Possible inputs:
  - Name / type / aspects / mystery tags
  - Vault information (vault + peril / guardian / curse, etc.)
  - Short specs of skills / incidents / visitors

- Outputs:
  - Full item descriptions
  - Scene text for expedition start / success
  - Short quest or incident blurbs
  - Worldbuilding lore paragraphs

### 📖 RAG / Knowledge Graph (Planned)

- Integrate BOH/CS lore and extracted structured data
- Use vector search and a light-weight knowledge graph to supply context to the model
- Reduce OOC (out-of-character / out-of-lore) outputs and keep generations consistent with the established worlds

---

## 📂 Project Structure 

> Actual structure may differ slightly in your repository; this is a typical layout.

```text
data/
  boh_raw_data.json         # Grouped raw data extracted from BOH
  cs_raw_data.json          # Grouped raw data extracted from CS
  train.jsonl               # Training set (instruction / input / output)
  eval.jsonl                # Evaluation set(planning)

scripts/
  boh_extractor.py          # Extract BOH elements / visitors / incidents / weather
  cs_extractor.py           # Extract CS books / tools / vaults / fragments / influences
  build_train_data.py       # Build training samples (translation + structured→text)
  train_qlora_qwen3.py      # QLoRA fine-tuning of Qwen3-0.6B
  infer_qlora_qwen3.py      # Inference script loading the LoRA adapter

models/
  qwen3_0_6b_qlora/         # Fine-tuned LoRA weights (base model not included)

README.md                   # This file
```

# 闰识书斋(中文)

闰识书斋（Numen Scriptorium）是一个基于 **Qwen3-0.6B** 微调的中英双语文本引擎，带有浓厚的密教、图书馆与世界观设定气息。

项目以《司辰之书》（Book of Hours）与《密教模拟器》（Cultist Simulator）的文本为灵感来源，通过提取和重构数据，训练出一个能写“同世界观风格”文案、并进行中英互译的模型，可用于：

- 物品描述（书籍、工具、材料、碎片、影响等）
- 探险地 / 事件 / 技能 / 访客的背景文字
- 世界观设定文档（lore）的撰写与扩展
- MOD 文本、本地化原型与同人创作

> ⚠️ **项目状态 / Project Status**  
> 该项目正处于开发阶段，目前用于微调的训练集仅包括《司辰之书》中书籍（tomes/journals 等）的文本信息。  
> 后续会逐步加入更多类型（技能、事件、访客、天气等）以及《密教模拟器》的相关数据。
> ⚠️ 本项目仅用于技术研究和个人创作实验，不包含或分发任何原始游戏资源。  
> 使用者需自行确保对游戏文本等数据的使用符合相关游戏的 EULA 与版权要求。

---

## ✨ 功能 Features

### 🧠 模型与微调

- 基于 **Qwen3-0.6B** 作为基础模型  
- 使用 **QLoRA（4bit 量化 + LoRA）** 在消费级 GPU（如 RTX 3080 8GB）上进行微调  
- 聚焦于：
  - 中↔英翻译，保持原作风格  
  - 按结构信息（名称、类型、aspects、mystery 等）生成完整物品描述与故事  
  - 输出稳定的中英双语对照文本

### 📚 数据来源与抽取

- **Book of Hours（司辰之书）**
  - 抽取：tomes、journals、aspecteditems、skills、incidents、abilities、visitors、weather 等
  - 解析 `xexts` / `xtriggers` 中的阅读文本与结果，构建阅读内容与学习结果结构
- **Cultist Simulator（密教模拟器）**
  - 抽取：books、tools、ingredients、fragments、influences、vaults 等
  - 从 recipes 中提取探险地（vaults）的不同阶段描述（开始/成功等）

所有数据先保存在 **分组的 raw JSON** 中，再由训练数据构建脚本转换为统一的 instruction-style 样本。

### 🌐 中英翻译与文风模仿

- 英文 → 中文：  
  - 保留原本设定信息，仿照 BOH/CS 的叙述方式（隐喻、宗教神话、仪式感）  
- 中文 → 英文：  
  - 输出接近 Lovecraftian / occult / gothic 文风的英文文本  
- 支持“先中文后英文”的双语对照输出，用于：

  - MOD 提示文本  
  - 设定书条目  
  - 世界观百科

### 🧩 结构化 → 文本生成

- 输入可以是：

  - 名称 / type / aspects / mystery 标签  
  - 探险地信息（vault + peril/guardian/curse 等）  
  - 技能/事件/访客的简要设定

- 输出：

  - 完整物品描述  
  - 探险开始/成功的情境文本  
  - 任务/事件简介  
  - 世界观设定段落

### 📖 RAG / 知识图谱（规划中）

- 整合 BOH / CS 的设定文档与提取的信息  
- 利用向量检索和轻量知识图谱，为模型提供上下文  
- 减少 OOC（设定崩坏），生成“符合既有世界观”的新文本

---

## 📂 项目结构 Project Structure

```text
data/
  boh_raw_data.json         # BOH 提取后的原始分组数据
  cs_raw_data.json          # CS 提取后的原始分组数据
  train.jsonl               # 训练集（instruction / input / output）
  eval.jsonl                # 验证集(计划中)

scripts/
  boh_extractor.py          # 提取 BOH 元素 / 访客 / 事件 / weather 等
  cs_extractor.py           # 提取 CS 书籍 / 工具 / 探险地 / 碎片 / 影响等
  build_train_data.py       # 从 raw_data 构建训练样本（翻译 + 结构→文本）
  train_qlora_qwen3.py      # 使用 QLoRA 微调 Qwen3-0.6B
  infer_qlora_qwen3.py      # 加载 LoRA 适配器进行推理

models/
  qwen3_0_6b_qlora/         # 微调后的 LoRA 权重（不包含原始基座模型）

README.md                   # 本文件
