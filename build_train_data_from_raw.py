import json
import os
from pathlib import Path
import random

RAW_FILE = os.path.join("data", "boh_raw_data_items_core.json")   # 你的原始文件
OUT_FILE = os.path.join("data", "train_items_core.jsonl")         # 输出训练数据

# 目标：结构化→文本样本在全体样本中的占比
TARGET_STRUCT_RATIO = 0.2   # 可调：0.1 ~ 0.3 之间都可以

# ====== 类型映射表（可以按需要继续补充） ======
TYPE_MAP_EN = {
    "书籍": "book",
    "信件": "letter",
    "卷轴": "scroll",
    "抄本": "codex",
    "石板": "tablet",
    "日记": "journal",
    "胶卷": "film",
    "唱片": "record",
}

TYPE_MAP_ZH = {
    "book": "书籍",
    "letter": "信件",
    "scroll": "卷轴",
    "codex": "抄本",
    "tablet": "石板",
    "journal": "日记",
    "film": "胶卷",
    "record": "唱片",
}

# ====== mystery 语义映射 ======
MYSTERY_ZH = {
    "lantern": "灯：关于不仁、理性、求知、辉光、启明、预知的准则。",
    "forge": "铸：关于转变、技巧、火、毁灭、塑形、力量、科技的准则。",
    "edge": "刃：关于蛮力、背叛、狡诈、痛苦、征服、统治、斗争、抗击的准则。",
    "winter": "冬：关于死亡、消逝、铭记、死寂、破败、雪、终末、结尾的准则。",
    "heart": "心：关于生命、存续、保护、永不停息的鼓点和舞蹈的准则。",
    "moth": "蛾：关于变化、奇想、非理性、直觉、寻觅、混沌、渴慕、激情、自然、林地、遗忘的准则。",
    "grail": "杯：关于感官欲望、生育、诱惑、苦痛、血与渴求、干渴与饥饿、贪婪、迷醉、愉悦的准则。",
    "knock": "启：关于伤口、锁匠、门与钥匙、揭示、洞开与拆解的准则。",
    "secrethistories": "秘史：关于被压制、遗忘、抹除、传说的准则。",
    "rose": "引：关于探索、启迪、希望。指引一切的罗盘玫瑰。通向新视界的九重引导。",
    "moon": "月：关于行于夜者，被遗忘者。秘密轻柔，夜柔尤甚；大海低语，而倾听未必总是明智。",
    "nectar": "蜜：很久以前，一些人把该准则称作“血”。世界脉络中的常绿珍宝；时节轮转的跃动脉搏。",
    "sky": "穹：关于平衡，和谐与必需。轻风，暴风，回响，歌咏；数学的复杂，飞行的原理。法则的碰触有时比我们想象的要轻。",
    "scale": "鳞：关于大地深处原始力量的残余。坚于表，固于里；难唤醒，更难抑。",
}

MYSTERY_EN = {
    "lantern": (
        "Lantern: precepts of unkindness, reason, knowledge,the Glory,lightenment, "
        "revelation and foresight."
    ),
    "forge": (
        "Forge: precepts of  transformation, artifice, fire, destruction, shaping, strength "
        "and technic power."
    ),
    "edge": (
        "Edge: precepts of  violence, betrayal, cunning, pain, conquest, "
        "dominion, struggle and resistance."
    ),
    "winter": (
        "Winter: precepts of death, passing and remembrance, stillness, decay, snow, "
        "endings and final silence."
    ),
    "heart": (
        "Heart: precepts of life, preservation, protection, and the drumbeat and dance that must never cease. ."
    ),
    "moth": (
        "Moth: precepts of change, whimsy, unreason, intuition, seeking, chaos, "
        "yearning, passion, nature, the Wood and forgetting."
    ),
    "grail": (
        "Grail: precepts of sensual desire and birth, seduction, pain, "
        "blood and thirst, hunger, greed, intoxication and delight."
    ),
    "knock": (
        "Knock: precepts of wounds, locksmiths, doors and keys, revelation, "
        "opening and unmaking."
    ),
    "secrethistories": (
        "Secrethistories: precepts of histories suppressed, forgotten or erased; of "
        "legends and their secret continuance."
    ),
    "rose": (
        "Rose: precepts of Exploration, Enlightenment, Hope; "
        "The rose which encompasseth all'. Nine directions to new horizons."
    ),
    "moon": (
        "Moon: precepts of the nocturnal, the forgotten; "
        "Secrets are soft; night is softer still; the sea speaks. It is not always wise to listen."
    ),
    "nectar": (
        "Nectar: Long ago, some called this principle Blood. "
        "The green wealth in the world's veins; the pulse of the seasons."
    ),
    "sky": (
        "Sky: Matters of balance, harmony and necessity."
        "Wind, storm, echo, song; the intricacies of mathematics and the principles of flight."
    ),
    "scale": (
        "Scale: precepts of what is left of the crude powers of the deep earth;"
        "Hard without, hard within, hard to rouse, harder to subdue."
    ),
}


def guess_type(item):
    """
    根据 item 的原始 type、id 和 aspects 中的 'other' 进行类型推断，
    尽量区分：book / letter / scroll / codex / tablet 等。
    """
    raw_type = (item.get("type") or "").strip().lower()
    _id = (item.get("id") or "").lower()
    aspects = item.get("aspects", {}) or {}

    # 先按 id 前缀 / 关键词做一些启发式判断
    if _id.startswith("letter.") or "letter" in _id:
        t = "letter"
    elif "scroll" in _id:
        t = "scroll"
    elif "tablet" in _id:
        t = "tablet"
    elif "codex" in _id:
        t = "codex"
    else:
        # 如果 id 没给出信息，再看 aspects 中的 'other' 类型线索
        other_keys = set()
        for k in aspects.keys():
            if "." not in k:
                other_keys.add(k)
            elif k.split(".", 1)[0] == "record":
                other_keys.add("record")

        # 排除无关状态型 key
        ignore = {"soph", "infinitereadable", "soaked", "readable"}
        other_keys = {x for x in other_keys if x not in ignore}

        # 按优先顺序判断：codex / scroll / tablet / journal / correspondence / invitation / delivery / film / record...
        if "codex" in other_keys:
            t = "codex"
        elif "scroll" in other_keys:
            t = "scroll"
        elif "tablet" in other_keys:
            t = "tablet"
        elif "journal" in other_keys:
            t = "journal"
        elif (
            "correspondence" in other_keys
            or "invitation" in other_keys
            or "delivery" in other_keys
        ):
            t = "letter"
        elif "film" in other_keys:
            t = "film"
        elif "record" in other_keys:
            t = "record"
        else:
            # 如果还是没有，就回退到原始 type
            t = raw_type or "item"

    type_zh = TYPE_MAP_ZH.get(t, "物品")
    return t, type_zh


# 一些风格提示模板，用来随机选择，避免 instruction 太单调
EN_ITEM_TEMPLATES = [
    "Translate the following Chinese item description into English, preserving the solemn, ritualistic tone and occult atmosphere.",
    "Translate the following Chinese item description into English, keeping a sombre, quasi-religious and slightly Lovecraftian style.",
]

ZH_ITEM_TEMPLATES = [
    "将下面的英文物品描述翻译为中文，保持阴郁庄严、带有神秘主义与宗教神话色彩的文风。",
    "将下面的英文物品描述翻译为中文，保持类似克苏鲁与哥特式的神秘氛围与仪式感。",
]

ZH_STORY_TEMPLATES = [
    "将下面的英文故事段落翻译为中文，保持《司辰之书》式的叙事风格，善用象征与暗示。",
    "将下面的英文故事段落翻译为中文，保留其谜语般隐喻与宗教神话气息。",
]

EN_STORY_TEMPLATES = [
    "Translate the following Chinese passage into English, preserving the dark, mythic and slightly occult narrative tone.",
    "Translate the following Chinese passage into English, keeping a gothic, symbol-laden style reminiscent of esoteric literature.",
]

ZH_MISC_TEMPLATES = [
    "将下面的英文文本翻译为中文，保持原作的神秘主义与哲思感，并保留隐喻与象征的味道。",
    "将下面的英文文本翻译为中文，保留其冷静而诡异的叙述方式与宗教神话暗流。",
]

EN_MISC_TEMPLATES = [
    "Translate the following Chinese passage into English, keeping the same sense of mystery, philosophical depth and occult symbolism.",
    "Translate the following Chinese passage into English, preserving its restrained but unsettling, myth-tinged tone.",
]


def add_example(examples, instruction, inp, out):
    """简单包装一条样本；如果 input / output 为空就忽略。"""
    if not inp or not out:
        return
    examples.append(
        {
            "instruction": instruction,
            "input": inp,
            "output": out,
        }
    )


def looks_untranslated(name: str, name_cn: str) -> bool:
    """判断 name_cn 是否基本没翻译（简单启发式）。"""
    if not name or not name_cn:
        return True
    n = name.strip().lower()
    c = name_cn.strip().lower()
    if n == c:
        return True
    # 去掉常见的引号、书名号等再比一次
    for ch in ["《", "》", '"', "“", "”", "'"]:
        n = n.replace(ch, "")
        c = c.replace(ch, "")
    return n == c


def aspects_to_tags(aspects: dict):
    """
    原始标签提取（保留），目前未在结构输入中直接使用，
    但可以作为调试/分析用途。
    """
    if not aspects:
        return [], []
    keys = list(aspects.keys())
    zh_tags = keys
    en_tags = [k.split(".")[-1] for k in keys]
    return zh_tags, en_tags


def aspects_to_semantic_text(aspects: dict, lang: str = "zh") -> str:
    """
    根据 aspects 生成“设定标签说明”，当前只处理 mystery.*，
    以后可以扩展 w / r / memories 等。
    """
    if not aspects:
        return ""

    lines = []

    # 1. mystery.*
    for k in aspects.keys():
        if k.startswith("mystery."):
            _, name = k.split(".", 1)
            if lang == "zh":
                desc = MYSTERY_ZH.get(name)
                if desc:
                    lines.append(f"准则:{desc}")
            else:
                desc = MYSTERY_EN.get(name)
                if desc:
                    lines.append(f"Principle: {desc}")

    return "\n".join(lines)


def main():
    raw_path = Path(RAW_FILE)
    assert raw_path.exists(), f"{RAW_FILE} 不存在"

    with raw_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("items", [])

    base_examples = []    # 翻译 / 对照类等“基础”样本
    struct_examples = []  # 结构化信息 -> 文本 的样本

    aspects_diff = {"other": set()}

    for item in items:
        name = (item.get("name") or "").strip()
        name_cn = (item.get("name_cn") or "").strip()
        desc = (item.get("description") or "").strip()
        desc_cn = (item.get("description_cn") or "").strip()
        aspects = item.get("aspects", {}) or {}

        readings = (item.get("readings", [])) or []
        readings_cn = (item.get("readings_cn", [])) or []

        results = (item.get("results", [])) or []
        
        if readings:
            if readings[0].get('intro') and readings[0].get('content'):
                readings = readings[0]['intro'].strip() + "\n " + readings[0]['content'].strip()
            elif readings[0].get('intro'):
                readings = readings[0]['intro'].strip()
            elif readings[0].get('content'):
                readings = readings[0]['content'].strip()

        if readings_cn:
            if readings_cn[0].get('intro') and readings_cn[0].get('content'):
                readings_cn = readings_cn[0]['intro'].strip() +"\n "+ readings_cn[0]['content'].strip()
            elif readings_cn[0].get('intro'):
                readings_cn = readings_cn[0]['intro'].strip()
            elif readings_cn[0].get('content'):
                readings_cn = readings_cn[0]['content'].strip()

        if desc.endswith(']'):
            if '<i>numen</i>' in desc:
                desc = desc.rsplit('[', 1)[0].strip()
        if desc_cn.endswith(']'):
            if '<i>闰识</i>' in desc_cn:
                desc_cn = desc_cn.rsplit('[', 1)[0].strip()

        if results:
            results_en = []
            results_cn = []
            results_en.append(results[0]['result_name'])
            results_cn.append(results[0]['result_name_cn'])
            results_en = ",".join(results_en)
            results_cn = ",".join(results_cn)

        # 统计 aspects 分布
        # for k in aspects.keys():
        #     if "." in k:
        #         key, value = k.split(".", 1)
        #         if key in aspects_diff:
        #             aspects_diff[key].add(value)
        #         else:
        #             aspects_diff[key] = set([value])
        #     else:
        #         aspects_diff["other"].add(k)

        type_en, type_zh = guess_type(item)
        zh_tags, en_tags = aspects_to_tags(aspects)

        # 语义标签文本
        semantic_zh = aspects_to_semantic_text(aspects, lang="zh")
        semantic_en = aspects_to_semantic_text(aspects, lang="en")

        # ========== 1. 物品描述：英 <-> 中（基础样本） ==========
        if readings:
            desc = desc+ "\n " + readings
            desc_cn = desc_cn+ "\n " + readings_cn

        if desc and desc_cn:
            # 名称如果没翻译，不强制要求用 name_cn
            if not looks_untranslated(name, name_cn):
                eng_name = name
                cn_name = name_cn
            else:
                eng_name = name
                cn_name = name  # 或者 ""，看你喜好

            eng_block = (
                f"Name: {eng_name}\n"
                f"Type: {type_en}\n"
                f"Description: {desc}"
            )
            cn_block = (
                f"名称：{cn_name}\n"
                f"类型：{type_zh}\n"
                f"描述：{desc_cn}"
            )

            # 1-1 英 -> 中
            zh_item_instr = random.choice(ZH_ITEM_TEMPLATES)
            add_example(
                base_examples,
                zh_item_instr,
                eng_block,
                cn_block,
            )

            # 1-2 中 -> 英
            en_item_instr = random.choice(EN_ITEM_TEMPLATES)
            add_example(
                base_examples,
                en_item_instr,
                cn_block,
                eng_block,
            )

            # 1-3 中英对照输出
            out_block = f"【中文】\n{cn_block}\n\n【English】\n{eng_block}"
            add_example(
                base_examples,
                "根据下面的英文物品描述，生成一个中英文对照版本。先输出中文的【名称】【类型】【描述】，再输出对应的英文【Name】【Type】【Description】。",
                eng_block,
                out_block,
            )

            # ========== 3. 结构化信息 -> 文本（结构样本） ==========
            # 中文方向：结构 -> 中文描述
            struct_input_zh = (
                "物品信息：\n"
                f"名称：{name_cn or name}\n"
                f"类型：{type_zh}\n"
                f"{semantic_zh if semantic_zh else ''}\n\n"
                f"习得：{results_cn if results_cn else ''}\n"
                "请根据以上信息，用中文写出该物品的详细描述，"
                "风格接近《司辰之书》，带有神秘主义、宗教神话与隐喻感。"
            )
            add_example(
                struct_examples,
                "根据给定的物品结构化信息，生成一段完整的中文物品描述。",
                struct_input_zh,
                desc_cn,
            )

            # 英文方向：结构 -> 英文描述
            struct_input_en = (
                "Item Info:\n"
                f"Name: {name}\n"
                f"Type: {type_en}\n"
                f"{semantic_en if semantic_en else ''}\n\n"
                f"Learned: {results_en if results_en else ''}\n"
                "Based on the information above, write a full English item description "
                "in the style of 'Book of Hours', with occult, quasi-religious and symbolic overtones."
            )
            add_example(
                struct_examples,
                "Given structured item information, generate a complete English item description.",
                struct_input_en,
                desc,
            )

        # ========== 2. readings / readings_cn 段落翻译（基础样本） ==========
        readings = item.get("readings", []) or []
        readings_cn = item.get("readings_cn", []) or []

        for r_en, r_cn in zip(readings, readings_cn):
            intro_en = (r_en.get("intro") or "").strip()
            content_en = (r_en.get("content") or "").strip()
            intro_cn = (r_cn.get("intro") or "").strip()
            content_cn = (r_cn.get("content") or "").strip()

            # 2-1 intro: 两端都有才用
            if intro_en and intro_cn:
                add_example(
                    base_examples,
                    random.choice(ZH_STORY_TEMPLATES),
                    intro_en,
                    intro_cn,
                )
                add_example(
                    base_examples,
                    random.choice(EN_STORY_TEMPLATES),
                    intro_cn,
                    intro_en,
                )

            # 2-2 content: 两端都有才用
            if content_en and content_cn:
                add_example(
                    base_examples,
                    random.choice(ZH_MISC_TEMPLATES),
                    content_en,
                    content_cn,
                )
 
                add_example(
                    base_examples,
                    random.choice(EN_MISC_TEMPLATES),
                    content_cn,
                    content_en,
                )

    # ====== 控制结构化样本占比 ======
    N_base = len(base_examples)
    N_struct_raw = len(struct_examples)

    # 目标：N_struct / (N_base + N_struct) ≈ TARGET_STRUCT_RATIO
    # 推出：N_struct_target = (TARGET_STRUCT_RATIO / (1 - TARGET_STRUCT_RATIO)) * N_base
    max_struct = int(TARGET_STRUCT_RATIO * N_base / (1.0 - TARGET_STRUCT_RATIO))

    if N_struct_raw > max_struct:
        struct_examples = random.sample(struct_examples, max_struct)
        print(
            f"结构化样本过多，已从 {N_struct_raw} 条随机抽取 {max_struct} 条，"
            f"目标占比约 {TARGET_STRUCT_RATIO:.2f}"
        )
    else:
        real_ratio = N_struct_raw / (N_base + N_struct_raw + 1e-9)
        print(
            f"结构化样本数量 {N_struct_raw} 条，小于目标上限 {max_struct} 条，"
            f"实际占比约 {real_ratio:.2f}"
        )

    all_examples = base_examples + struct_examples

    # ========== 写出 jsonl ==========
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"基础样本 {N_base} 条，结构化样本 {len(struct_examples)} 条，"
        f"总计 {len(all_examples)} 条，已写入 {OUT_FILE}"
    )


if __name__ == "__main__":
    main()