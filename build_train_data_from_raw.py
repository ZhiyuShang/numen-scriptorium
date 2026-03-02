import json
import os
from pathlib import Path
import random
import hashlib
# import zhconv
# import re

# try:
#     from translate import Translator
#     # 初始化英译中翻译器
#     _translator = Translator(from_lang="en", to_lang="zh-CN")
# except ImportError:
#     print("未检测到 translate 库，自动翻译功能将回退为默认'物品'。可运行 pip install translate 安装。")
#     _translator = None
# _auto_trans_cache = {}

RAW_FILE = os.path.join("data", "boh_raw_data_items.json")  # 你的原始文件
OUT_FILE = os.path.join("data", "train_items_boh.jsonl")    # 输出训练数据

GAME = "司辰之书" if 'boh' in RAW_FILE else "密教模拟器" 

# 目标：结构化→文本样本在全体样本中的占比
TARGET_STRUCT_RATIO = 0.2   # 可调：0.1 ~ 0.3 之间都可以

# 内层 *_cn 对文本的最小长度（太短的字段通常是噪声，例如 label/title/单词）
MIN_NESTED_PAIR_LEN = 0

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
    "密传": "fragment",
    "影响": "influence",
    "工具": "tool",
    "原料": "ingredient",
    "访客": "visitor",
    "天气": "weather",
    "魂质": "ability",
    "事件": "incident",
    "技艺": "skills"
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
    "fragment":"密传",
    "influence":"影响",
    "tool": "工具",
    "ingredient":"原料",
    'visitor':'访客',
    'weather':'天气',
    'ability':'魂质',
    'incident':'事件',
    'skills': '技艺',
    'skills_r':'技艺',
    # === 根据 EDA 频次表精选的 AspectedItem 高频前缀 ===
    "painting": "画作",
    "bust": "半身像",
    "bottle": "瓶装物",
    "pot": "壶",
    "chair": "椅子",
    "mem": "回忆",
    "weapon": "武器",
    "candle": "蜡烛",
    "numen": "闰识",
    "pie": "派",
    "casket": "箱匣",
    "glass": "杯装饮品",
    "tea": "茶",
    "paint": "颜料",
    "lamp": "灯具",
    "sofa": "沙发",
    "stool": "凳子",
    "vase": "花瓶",
    "veg": "蔬菜",
    "armchair": "扶手椅",
    "clock": "钟表",
    "cup": "杯装饮品",
    "egg": "蛋",
    "wire": "金属丝",
    "music": "乐曲",
    "mirror": "镜子",
    "liquid": "液体",
    "packet": "包装冲剂",
    "spintria": "古币",
    "cake": "蛋糕",
    "mackerel": "鲭鱼",
    "soup": "汤",
    "statuette": "小雕像",
    "chicken": "家禽",
    "sacrament": "圣餐",
    "statue": "雕像",
    "marrow": "西葫芦",
    "meringue": "蛋白酥",
    "hours": "司辰",      # 原作中特指司辰题材的物品/雕像
    "stag": "鹿",
    "window": "花窗玻璃",
    "basket": "筐",
    "dog": "狗",
    "gull": "海鸥",
    "cat": "猫",
    "viper": "蛇",
    "assam": "茶",         # 阿萨姆统一归为茶
    "lapsang": "茶",       # 正山小种统一归为茶
    "box": "盒子",
    "packages": "包裹",
    "bucket": "桶",
    "footstool": "脚凳",
    "bolt": "布匹",
    "beef": "牛肉",
    "oil": "油",
    "bread": "面包",
    "pheasant": "野鸡",
    "pumpkin": "南瓜",
    "toastie": "吐司",
    "shield": "盾徽",
    "weapons": "武器",
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

SPRITE_ZH_MAP = {
    "lantern": "灯", "forge": "铸", "edge": "刃", "winter": "冬",
    "heart": "心", "moth": "蛾", "grail": "杯", "knock": "启",
    "secrethistories": "秘史", "rose": "引", "moon": "月",
    "nectar": "蜜", "sky": "穹", "scale": "鳞"
}

# ====== 语言学名映射 (专门用于解析 CS 中的 scholar 产出) ======
SCHOLAR_LANG_DICT_EN = {
    "fucine": "Fucine",
    "phrygian": "Phrygian",
    "vak": "Vak",
    "aramaic": "Aramaic",
    "greek": "Greek",
    "latin": "Latin",
    "sanskrit": "Sanskrit",
    "mandaic":"Mandaic"
}

SCHOLAR_LANG_DICT_ZH = {
    "fucine": "富奇诺语",
    "phrygian": "弗里吉亚语",
    "vak": "伐诃语",
    "aramaic": "亚兰语",
    "greek": "希腊语",
    "latin": "拉丁语",
    "sanskrit": "梵语",
    "mandaic":"曼达安语"
}


def guess_type(item):
    """
    根据 item 的原始 type、id 和 aspects 进行类型推断。
    使用绝对安全的白名单机制，无法精准确定的类别一律回退为通用的“物品”。
    """
    raw_type = (item.get("type") or "").strip().lower()
    _id = (item.get("id") or "").lower()
    aspects = item.get("aspects", {}) or {}

    # 1. 基于 ID 强特征的推断
    if _id.startswith("letter.") or "letter" in _id:
        t = "letter"
    elif "scroll" in _id:
        t = "scroll"
    elif "tablet" in _id:
        t = "tablet"
    elif "codex" in _id:
        t = "codex"
    else:
        # 2. 基于 Aspects 包含的特殊性相推断
        other_keys = set()
        for k in aspects.keys():
            if "." not in k:
                other_keys.add(k)
            elif k.split(".", 1)[0] == "record":
                other_keys.add("record")

        ignore = {"soph", "infinitereadable", "soaked", "readable"}
        other_keys = {x for x in other_keys if x not in ignore}

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
            # 3. 处理 aspecteditem
            if raw_type == "aspecteditem":
                # 如果有 '.'，提取前缀
                if "." in _id:
                    prefix = _id.split(".", 1)[0]
                    if prefix in TYPE_MAP_ZH:
                        t = prefix
                    else:
                        t = "item"
                # 如果没有 '.'，信息量不足，直接舍去强行分类
                else:
                    t = "item"
            else:
                t = raw_type or "item"

    # 4. 查表获取中文名，如果 t 完全不认识，则安全兜底为“物品”
    type_zh = TYPE_MAP_ZH.get(t, "物品")
    
    # 首字母大写处理英文名（针对兜底的 item，输出 Item）
    type_en = t.capitalize() if t else "Item"
    
    return type_en, type_zh


EN_ITEM_TEMPLATES = [
    "Translate the following Chinese item description into English, preserving the solemn, ritualistic tone and occult atmosphere.",
    "Translate the following Chinese item description into English, keeping a sombre, quasi-religious and slightly Lovecraftian style.",
]

ZH_ITEM_TEMPLATES = [
    "将下面的英文物品描述翻译为中文，保持阴郁庄严、带有神秘主义与宗教神话色彩的文风。",
    "将下面的英文物品描述翻译为中文，保持类似克苏鲁与哥特式的神秘氛围与仪式感。",
]

ZH_STORY_TEMPLATES = [
    f"将下面的英文故事段落翻译为中文，保持《{GAME}》式的叙事风格，善用象征与暗示。",
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

ZH_CONTINUATION_TEMPLATES = [
    f"根据以下探险地点的背景和初始情况，续写接下来的探险过程与最终发现。保持《{GAME}》那种阴暗、神秘且带有暗示的文风。",
    "你正在参与一场秘密的神秘学探险。基于提供的初始情境，续写突破阻碍后的场景以及最终的结局。"
]

EN_CONTINUATION_TEMPLATES = [
    "Continue the narrative of this secret expedition based on the initial setup. Preserve the dark, mysterious, and occult tone of 'Cultist Simulator'.",
    "Based on the given expedition premise, write the continuation of the story, describing the breakthrough and the final discoveries."
]

def add_example(examples, instruction, inp, out, seen=None):
    """简单包装一条样本；如果 input / output 为空就忽略。支持去重。"""
    if not inp or not out:
        return
    ex = {"instruction": instruction, "input": inp, "output": out}
    if seen is not None:
        sig = hashlib.md5(
            (instruction + "\n" + inp + "\n" + out).encode("utf-8", errors="ignore")
        ).hexdigest()
        if sig in seen:
            return
        seen.add(sig)
    examples.append(ex)


def looks_untranslated(name: str, name_cn: str) -> bool:
    if not name or not name_cn:
        return True
    n = name.strip().lower()
    c = name_cn.strip().lower()
    if n == c:
        return True
    for ch in ["《", "》", '"', "“", "”", "'"]:
        n = n.replace(ch, "")
        c = c.replace(ch, "")
    return n == c


def aspects_to_semantic_text(aspects: dict, lang: str = "zh") -> str:
    if not aspects:
        return ""
    lines = []
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
        elif k in MYSTERY_EN.keys():
            if lang == "zh":
                lines.append(f"准则:{MYSTERY_ZH[k]}")
            else:
                lines.append(f"Principle: {MYSTERY_EN[k]}")
    return "\n".join(lines)

def is_boring_short_text(text: str) -> bool:
    """
    判断文本是否“短且无趣”。
    规则：如果长度小于 12 个字符，并且句中没有逗号、分号或冒号，则判定为过于简单的短句。
    """
    if len(text) < 12 and not any(punc in text for punc in ["，", ",", "；", ";", "：", ":"]):
        return True
    return False

def clean_text(text: str) -> str:
    """
    智能文本清洗函数：
    1. 剔除末尾的注记
    2. 将图标 <sprite name=xxx> 还原为对应文字
    3. 将 <b> 和 <i> 转换为大模型更易理解的 Markdown 语法
    """
    if not text:
        return text
    t = text.strip()
    
    # 1. 剔除末尾无意义的注记 [...]
    if t.endswith(".]") or t.endswith('。]') or t.endswith('.].'):
        t = t.rsplit("[", 1)[0].strip()
        
    # 2. 智能处理 <sprite> 图标标签
    # def sprite_repl(match):
    #     sprite_name = match.group(1).lower()
    #     # 简单判定：如果当前句子包含中文，则翻译为中文性相；否则首字母大写
    #     if any('\u4e00' <= char <= '\u9fff' for char in t):
    #         return SPRITE_ZH_MAP.get(sprite_name, sprite_name)
    #     else:
    #         return sprite_name.capitalize()
            
    # t = re.sub(r'<sprite\s+name=([^>]+)>', sprite_repl, t)
    
    # # 3. 将 <b> 和 <i> 转换为 Markdown 语法（大模型更敏感）
    # # 如果你想保留 HTML 给游戏 Mod 用，可以直接注释掉这两行
    # t = t.replace('<b>', '**').replace('</b>', '**')
    # t = t.replace('<i>', '*').replace('</i>', '*')
    
    # # 4. 兜底清洗：删掉其他任何不认识的异常 HTML 标签，但保留我们刚刚转换的文本
    # t = re.sub(r'<[^>]+>', '', t)
    
    return t.strip()


def first_reading_text(readings_list):
    """
    从 readings/readings_cn 的 list[dict] 里提取第一条的 intro/content 合并文本。
    返回字符串（可能为空）。
    """
    if not readings_list or not isinstance(readings_list, list):
        return ""
    r0 = readings_list[0] if readings_list else None
    if not isinstance(r0, dict):
        return ""
    intro = (r0.get("intro") or "").strip()
    content = (r0.get("content") or "").strip()
    if intro and content:
        return intro + "\n " + content
    return intro or content


def main():
    raw_path = Path(RAW_FILE)
    assert raw_path.exists(), f"{RAW_FILE} 不存在"

    with raw_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("items", [])

    item_id_map_en = {it.get("id"): it.get("name") for it in items if "id" in it}
    item_id_map_zh = {it.get("id"): it.get("name_cn") for it in items if "id" in it}

    base_examples = []
    struct_examples = []

    # 去重集合：避免同一条样本重复写入
    seen_base = set()
    seen_struct = set()

    for item in items:
        name = (item.get("name") or "").strip()
        name_cn = (item.get("name_cn") or "").strip()

        desc = (item.get("description") or "").strip()
        desc_cn = (item.get("description_cn") or "").strip()

        # 清理末尾 [] 注记
        desc = clean_text(desc) #"<i>numen</i>" "<i>闰识</i>"
        desc_cn = clean_text(desc_cn)

        aspects = item.get("aspects", {}) or {}

        # results（只取第一条，保持和你原来一致）
        results = (item.get("results", []) or [])
        results_en = ""
        results_cn = ""
        if results and isinstance(results, list) and isinstance(results[0], dict):
            results_en = (results[0].get("result_name") or "").strip()
            results_cn = (results[0].get("result_name_cn") or "").strip()
        elif item.get("type") == "book" and "effects" in item:
            # CS 逻辑：只有类别为 book 且含有 effects 时才提取“习得”
            effects = item.get("effects", {})
            for eff_key, eff_val in effects.items():
                if eff_val > 0:  # 大于 0 才是读书产出物 (消耗品通常是 -1)
                    if eff_key.startswith("fragment"):
                        results_en = item_id_map_en.get(eff_key, eff_key)
                        results_cn = item_id_map_zh.get(eff_key, eff_key)
                        break
                    elif eff_key.startswith("scholar"):
                        lang_key = eff_key.replace("scholar", "")
                        results_en = SCHOLAR_LANG_DICT_EN.get(lang_key, lang_key)
                        results_cn = SCHOLAR_LANG_DICT_ZH.get(lang_key, lang_key)
                        break

        # readings：用“单独变量”提取第一条，避免污染后面的 list
        reading_text_en = first_reading_text(item.get("readings", []) or [])
        reading_text_cn = first_reading_text(item.get("readings_cn", []) or [])

        # 只有两边都有，才拼进 desc，保证对齐
        if reading_text_en and reading_text_cn:
            desc = (desc + "\n " + reading_text_en).strip()
            desc_cn = (desc_cn + "\n " + reading_text_cn).strip()

        type_en, type_zh = guess_type(item)

        semantic_zh = aspects_to_semantic_text(aspects, lang="zh")
        semantic_en = aspects_to_semantic_text(aspects, lang="en")

        # ========== 1. 物品描述：英 <-> 中（基础样本） ==========
        if desc and desc_cn:
            if not looks_untranslated(name, name_cn):
                eng_name = name
                cn_name = name_cn
            else:
                eng_name = name
                cn_name = name

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

            add_example(
                base_examples,
                random.choice(ZH_ITEM_TEMPLATES),
                eng_block,
                cn_block,
                seen=seen_base,
            )
            add_example(
                base_examples,
                random.choice(EN_ITEM_TEMPLATES),
                cn_block,
                eng_block,
                seen=seen_base,
            )

            out_block = f"【中文】\n{cn_block}\n\n【English】\n{eng_block}"
            add_example(
                base_examples,
                "根据下面的英文物品描述，生成一个中英文对照版本。先输出中文的【名称】【类型】【描述】，再输出对应的英文【Name】【Type】【Description】。",
                eng_block,
                out_block,
                seen=seen_base,
            )

            # ========== 3. 结构化信息 -> 文本（结构样本） ==========
            if not is_boring_short_text(desc_cn):
                lines_zh = [
                    "物品信息：",
                    f"名称：{name_cn or name}",
                    f"类型：{type_zh}"
                ]
                if semantic_zh:
                    lines_zh.append(semantic_zh)
                if results_cn:  # 只有成功提取到了产出物，才会加入"习得："行
                    lines_zh.append(f"习得：{results_cn}")
                    
                struct_input_zh = (
                    "\n".join(lines_zh) + "\n\n"
                    "请根据以上信息，用中文写出该物品的详细描述，"
                    f"风格接近《{GAME}》，带有神秘主义、宗教神话与隐喻感。"
                )   
                add_example(
                    struct_examples,
                    "根据给定的物品结构化信息，生成一段完整的中文物品描述。",
                    struct_input_zh,
                    desc_cn,
                    seen=seen_struct,
                )
            if not is_boring_short_text(desc):
                lines_en = [
                    "Item Info:",
                    f"Name: {name}",
                    f"Type: {type_en}"
                ]
                if semantic_en:
                    lines_en.append(semantic_en)
                if results_en:  # 同理，仅当存在产出时加入"Learned:"行
                    lines_en.append(f"Learned: {results_en}")
                    
                struct_input_en = (
                    "\n".join(lines_en) + "\n\n"
                    "Based on the information above, write a full English item description "
                    "in the style of 'Book of Hours', with occult, quasi-religious and symbolic overtones."
                )
                add_example(
                    struct_examples,
                    "Given structured item information, generate a complete English item description.",
                    struct_input_en,
                    desc,
                    seen=seen_struct,
                )

        # ========== 2. readings / readings_cn 段落翻译（基础样本） ==========
        readings = item.get("readings", []) or []
        readings_cn = item.get("readings_cn", []) or []

        if isinstance(readings, list) and isinstance(readings_cn, list):
            for r_en, r_cn in zip(readings, readings_cn):
                if not isinstance(r_en, dict) or not isinstance(r_cn, dict):
                    continue

                intro_en = (r_en.get("intro") or "").strip()
                content_en = (r_en.get("content") or "").strip()
                intro_cn = (r_cn.get("intro") or "").strip()
                content_cn = (r_cn.get("content") or "").strip()

                if intro_en and intro_cn:
                    add_example(
                        base_examples,
                        random.choice(ZH_STORY_TEMPLATES),
                        intro_en,
                        intro_cn,
                        seen=seen_base,
                    )
                    add_example(
                        base_examples,
                        random.choice(EN_STORY_TEMPLATES),
                        intro_cn,
                        intro_en,
                        seen=seen_base,
                    )

                if content_en and content_cn:
                    add_example(
                        base_examples,
                        random.choice(ZH_MISC_TEMPLATES),
                        content_en,
                        content_cn,
                        seen=seen_base,
                    )
                    add_example(
                        base_examples,
                        random.choice(EN_MISC_TEMPLATES),
                        content_cn,
                        content_en,
                        seen=seen_base,
                    )
        # ========== 2.x 探险地 (vault) 特有字段翻译与故事续写 ==========
        vault_descriptions = item.get("descriptions", {})
        if isinstance(vault_descriptions, dict):
            # 1. 提取各个阶段的文本
            setup_en = (vault_descriptions.get("setup_start", {}).get("en") or "").strip()
            setup_cn = (vault_descriptions.get("setup_start", {}).get("cn") or "").strip()
            suc_start_en = (vault_descriptions.get("success_start", {}).get("en") or "").strip()
            suc_start_cn = (vault_descriptions.get("success_start", {}).get("cn") or "").strip()
            suc_desc_en = (vault_descriptions.get("success_desc", {}).get("en") or "").strip()
            suc_desc_cn = (vault_descriptions.get("success_desc", {}).get("cn") or "").strip()

            setup_en = clean_text(setup_en)
            setup_cn = clean_text(setup_cn)
            suc_start_en = clean_text(suc_start_en)
            suc_start_cn = clean_text(suc_start_cn)
            suc_desc_en = clean_text(suc_desc_en)
            suc_desc_cn = clean_text(suc_desc_cn)

            # 2. 原有的翻译样本对 
            for desc_key in ["setup_start", "success_start", "success_desc"]:
                desc_obj = vault_descriptions.get(desc_key)
                if isinstance(desc_obj, dict):
                    en_text = (desc_obj.get("en") or "").strip()
                    en_text = clean_text(en_text)
                    cn_text = (desc_obj.get("cn") or "").strip()
                    cn_text = clean_text(cn_text)
                    if en_text and cn_text:
                        add_example(base_examples, random.choice(ZH_STORY_TEMPLATES), en_text, cn_text, seen=seen_base)
                        add_example(base_examples, random.choice(EN_STORY_TEMPLATES), cn_text, en_text, seen=seen_base)

            clean_name_en = name
            if ":" in clean_name_en:
                clean_name_en = clean_name_en.split(":", 1)[1].strip()
            elif clean_name_en.startswith("An Expedition to"):
                clean_name_en = clean_name_en.replace("An Expedition to", "").strip()

            clean_name_cn = name_cn
            if "：" in clean_name_cn:  # 中文全角冒号
                clean_name_cn = clean_name_cn.split("：", 1)[1].strip()
            elif ":" in clean_name_cn: # 兼容半角冒号
                clean_name_cn = clean_name_cn.split(":", 1)[1].strip()
            elif clean_name_cn.startswith("探险"):
                clean_name_cn = clean_name_cn.replace("探险", "", 1).strip()

            # 3. 探险故事续写样本（中文）
            if setup_cn and suc_start_cn and suc_desc_cn:
                inp_cn = f"探险地：{clean_name_cn}\n背景：{setup_cn}"
                out_cn = f"{suc_start_cn}\n\n{suc_desc_cn}"
                add_example(
                    base_examples, 
                    random.choice(ZH_CONTINUATION_TEMPLATES), 
                    inp_cn, 
                    out_cn, 
                    seen=seen_base
                )

            # 4. 探险故事续写样本（英文）
            if setup_en and suc_start_en and suc_desc_en:
                inp_en = f"Expedition: {clean_name_en}\nPremise: {setup_en}"
                out_en = f"{suc_start_en}\n\n{suc_desc_en}"
                add_example(
                    base_examples, 
                    random.choice(EN_CONTINUATION_TEMPLATES), 
                    inp_en, 
                    out_en, 
                    seen=seen_base
                )

        # ========== 2.x 事件 (incident) 访客反应翻译 ==========
        visitacted_reactions = item.get("visitacted_reactions", {})
        if isinstance(visitacted_reactions, dict):
            # 遍历访客 (如 "arun", "connie")
            for visitor_id, aspect_dict in visitacted_reactions.items():
                if isinstance(aspect_dict, dict):
                    # 遍历对应的性相 (如 "moon", "knock")
                    for aspect, reaction_data in aspect_dict.items():
                        if isinstance(reaction_data, dict):
                            en_text = (reaction_data.get("description") or "").strip()
                            cn_text = (reaction_data.get("description_cn") or "").strip()
                            
                            # 清除可能存在的标签注记
                            en_text = clean_text(en_text)
                            cn_text = clean_text(cn_text)
                            
                            if en_text and cn_text:
                                # 访客的反应通常带有很强的叙事色彩与谜语人风格，
                                # 使用 MISC 模板（神秘主义/哲思感）或 STORY 模板最为合适
                                add_example(
                                    base_examples,
                                    random.choice(ZH_MISC_TEMPLATES),
                                    en_text,
                                    cn_text,
                                    seen=seen_base,
                                )
                                add_example(
                                    base_examples,
                                    random.choice(EN_MISC_TEMPLATES),
                                    cn_text,
                                    en_text,
                                    seen=seen_base,
                                )
    # ====== 控制结构化样本占比 ======
    N_base = len(base_examples)
    N_struct_raw = len(struct_examples)

    max_struct = int(TARGET_STRUCT_RATIO * N_base / (1.0 - TARGET_STRUCT_RATIO)) if N_base > 0 else 0

    if N_struct_raw > max_struct and max_struct > 0:
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
    out_path = Path(OUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(
        f"基础样本 {len(base_examples)} 条，结构化样本 {len(struct_examples)} 条，"
        f"总计 {len(all_examples)} 条，已写入 {OUT_FILE}"
    )


if __name__ == "__main__":
    main()