import json
import os
import time
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional

from google import genai
from google.genai import types
import re
# ========= 配置 =========

WIKI_URL = "https://mansus.huijiwiki.com/wiki/%E6%BC%AB%E5%AE%BF%E5%8E%86%E5%8F%B2"
OUTPUT_JSON = "mansus_history_events_rag.json"

os.environ.get("MY_API_KEY")
client = genai.Client()
GEMINI_MODEL = "gemini-2.5-flash"

# ========= 工具函数 =========


HTML_CACHE_PATH = "data/mansus_history.html"

def fetch_html(url: str) -> str:
    if os.path.exists(HTML_CACHE_PATH):
        with open(HTML_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://mansus.huijiwiki.com/",
        "Sec-CH-UA": "\"Not:A-Brand\";v=\"99\", \"Google Chrome\";v=\"145\", \"Chromium\";v=\"145\"",
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": "\"Windows\"",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
        "Cookie": "_ga=GA1.2.945804195.1766784272; _gid=GA1.2.804648546.1771717152; _ga_N3DS04643Q=GS2.2.s1771729275$o32$g0$t1771729275$j60$l0$h0; __cf_bm=E7OGtHcFAVkCI.1k3hjzSbDRB6A0KSVa8RuTooMzMn4-1771763188-1.0.1.1-8MA.Clfqc9jslfqOiZ2qKG7O_AyakHWIqiiQGAU8nGVBUIhAlNZ.3kwnAOR0N4Y1.FOU0fImTiRLJHbRbDpfoyDPzv4o4mxRNesjs2NJg44",
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    html = resp.text

    os.makedirs(os.path.dirname(HTML_CACHE_PATH) or ".", exist_ok=True)
    with open(HTML_CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    return html

def parse_article_structure(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="wiki-body-section", role="main")
    if not article:
        raise RuntimeError("Cannot find target <article> section.")

    data: Dict[str, Any] = {}
    
    # 1. 预设“引言”状态：这样在遇到第一个 <h2> 之前出现的所有 <p> 标签，
    # 都会被自动接住，并归类到“漫宿历史与时代划分”这个伪事件中。
    current_era = "引言"
    current_h3 = "漫宿历史与时代划分"
    current_h4 = None

    data[current_era] = {
        "title": current_era,
        "events": {
            current_h3: {
                "level": "h3",
                "paragraphs": [],
                "subevents": {}
            }
        }
    }

    # 开始遍历 DOM 树
    for el in article.descendants:
        if not getattr(el, "name", None):
            continue
        name = el.name.lower()

        if name == "h2":
            # 遇到新的 h2，切换时代
            current_era = el.get_text(strip=True)
            data.setdefault(current_era, {"title": current_era, "events": {}})
            current_h3 = None
            current_h4 = None

        elif name == "h3":
            if not current_era:
                continue
            current_h3 = el.get_text(strip=True)
            current_h4 = None
            data[current_era]["events"].setdefault(
                current_h3,
                {"level": "h3", "paragraphs": [], "subevents": {}}
            )

        elif name == "h4":
            if not current_era or not current_h3:
                continue
            current_h4 = el.get_text(strip=True)
            data[current_era]["events"][current_h3]["subevents"].setdefault(
                current_h4,
                {"level": "h4", "paragraphs": []}
            )

        elif name == "p":
            if not current_era or not current_h3:
                continue
                
            text = el.get_text(strip=True)
            if not text:
                continue
                
            text = re.sub(r'\[\d+\]', '', text)
            
            event_obj = data[current_era]["events"][current_h3]
            if current_h4:
                event_obj["subevents"][current_h4]["paragraphs"].append(text)
            else:
                event_obj["paragraphs"].append(text)

    # 2. 后置清理：遍历提取到的数据，剔除没有任何段落内容的“空壳”节点
    cleaned_data = {}
    for era, era_obj in data.items():
        valid_events = {}
        for h3_title, event_obj in era_obj["events"].items():
            has_h3_paras = len(event_obj["paragraphs"]) > 0
            
            # 顺便清理空的 h4 子事件
            valid_subevents = {}
            for h4_title, sub_obj in event_obj["subevents"].items():
                if len(sub_obj["paragraphs"]) > 0:
                    valid_subevents[h4_title] = sub_obj
            event_obj["subevents"] = valid_subevents
            
            # 只要 h3 自身有段落，或者其子节点 h4 有段落，就视为有效事件并保留
            if has_h3_paras or len(valid_subevents) > 0:
                valid_events[h3_title] = event_obj
                
        # 只要这个大时代 (h2) 下存在有效的事件，就保留整个大时代
        if len(valid_events) > 0:
            era_obj["events"] = valid_events
            cleaned_data[era] = era_obj
    return cleaned_data


def is_conflict_or_death_event(title: str, paragraphs: List[str]) -> bool:
    """
    粗略判断是否是“司辰斗争 / 死亡”相关重大事件，用于决定摘要长度。
    可以根据需要扩展关键词。
    """
    text = title + "\n" + "\n".join(paragraphs)
    keywords = [
        "覆石之战", "太阳大战", "大战", "战争",
        "被", "杀死", "斩杀", "粉碎", "饮干",
        "除名", "分裂", "死亡", "陨落",'毁灭','击败','猎杀'
    ]
    # 简单规则：出现“战”“大战”等高风险词，或者“被…杀死/斩杀”等
    for kw in keywords:
        if kw in text:
            return True
    return False


def summarise_event_text(
    era: str,
    title: str,
    paragraphs: List[str],
    is_conflict: bool
) -> str:
    full_text = "\n\n".join(paragraphs)

    if is_conflict:
        length_hint = "请写 4~6 句中文摘要，适当具体描述关键冲突、参与者与结果。"
    else:
        length_hint = "请写 2~4 句中文摘要，突出关键参与者、起因与后果。"

    # ========== 新增修改 ==========
    # 强化 Prompt，严禁直接摘抄，以绕过 Recitation 拦截
    system_prompt = (
        "你是一个世界观设定编辑，现在要为漫宿相关的历史事件生成适合 RAG 的精炼摘要。\n"
        "总体要求：\n"
        "1. 使用中文输出。\n"
        "2. 保持信息密度高，不写旁白、不写对白，不编造新设定。\n"
        "3. 尽量保留关键参与者（司辰/派系/起源）、事件起因与影响。\n"
        "4. 【极其重要】绝对不可使用引号原样摘抄原文的词句！必须完全使用你自己的语言进行转述（Paraphrase），否则会被判定为抄袭。\n"
    )

    user_prompt = (
        f"时代（h2）：{era}\n"
        f"事件标题：{title}\n\n"
        f"原始段落：\n{full_text}\n\n"
        f"{length_hint}"
    )

    # 放宽安全限制
    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=2048,
            safety_settings=safety_settings
        )
    )

    # ========== 新增修改 ==========
    # 强制诊断输出：如果文本被截断，告诉你到底是撞了什么拦截墙
    if resp.candidates:
        finish_reason = resp.candidates[0].finish_reason.name
        if finish_reason != "STOP":
            print(f"\n[拦截警告] 事件 '{title}' 被意外截断！原因代码: {finish_reason}")
            # 如果原因是 RECITATION，说明模型还是照抄了；如果是 SAFETY，说明还有别的敏感词。

    return resp.text.strip()

def build_rag_json(structured: Dict[str, Any]) -> Dict[str, Any]:
    """
    输出结构：
    {
      era_h2: {
        "title":...,
        "events": {
          h3_title: {
            "level": "h3",
            "paragraphs": [...],
            "summary_cn": "...",
            "subevents": {
              h4_title: {
                "level": "h4",
                "paragraphs": [...],
                "summary_cn": "..."
              }
            }
          }
        }
      }
    }
    """
    rag = {}

    for era, era_obj in structured.items():
        rag[era] = {"title": era_obj["title"], "events": {}}
        for h3_title, event_obj in era_obj["events"].items():
            paragraphs_h3 = event_obj.get("paragraphs", [])
            subevents = event_obj.get("subevents", {})

            # 先 summarise h3 主事件本身
            event_entry = {
                "level": "h3",
                "paragraphs": paragraphs_h3,
                "summary_cn": ""
            }
            if paragraphs_h3:
                is_conflict = is_conflict_or_death_event(h3_title, paragraphs_h3)
                try:
                    summary = summarise_event_text(era, h3_title, paragraphs_h3, is_conflict)
                    time.sleep(1.0)
                except Exception as e:
                    print(f"[WARN] summarise failed for {era} / {h3_title}: {e}")
                    summary = ""
                event_entry["summary_cn"] = summary

            # 再 summarise 每个 h4 子事件
            subevents_out = {}
            for h4_title, sub_obj in subevents.items():
                paras_h4 = sub_obj.get("paragraphs", [])
                if not paras_h4:
                    continue
                is_conflict_sub = is_conflict_or_death_event(h4_title, paras_h4)
                try:
                    summary_h4 = summarise_event_text(era, h4_title, paras_h4, is_conflict_sub)
                    time.sleep(1.0)
                except Exception as e:
                    print(f"[WARN] summarise failed for {era} / {h3_title} / {h4_title}: {e}")
                    summary_h4 = ""
                subevents_out[h4_title] = {
                    "level": "h4",
                    "paragraphs": paras_h4,
                    "summary_cn": summary_h4
                }

            event_entry["subevents"] = subevents_out
            rag[era]["events"][h3_title] = event_entry

    return rag


def main():
    print("[1] Fetching page...")
    html = fetch_html(WIKI_URL)

    print("[2] Parsing article structure (h2/h3/h4/p)...")
    structured = parse_article_structure(html)

    print("[3] Summarising events via Gemini (with conflict-aware length)...")
    rag_json = build_rag_json(structured)

    print(f"[4] Saving JSON to {OUTPUT_JSON}...")
    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rag_json, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()