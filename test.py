import os
import re
from google import genai
from google.genai import types

# 1. 配置你的 API Key
api_key = os.environ.get("MY_API_KEY")
if not api_key:
    raise ValueError("API Key 未找到，请检查环境配置！")
client = genai.Client(api_key=api_key)
breakpoint()
GEMINI_MODEL = "gemini-2.5-flash"

def summarise_event_text(era: str, title: str, paragraphs: list, is_conflict: bool) -> str:
    # 新增：在传入给大模型之前，先用正则把 [1], [2] 这种角标洗掉
    cleaned_paragraphs = [re.sub(r'\[\d+\]', '', p) for p in paragraphs]
    full_text = "\n\n".join(cleaned_paragraphs)

    if is_conflict:
        length_hint = "请写 4~6 句中文摘要，适当具体描述关键冲突、参与者与结果。"
    else:
        length_hint = "请写 2~4 句中文摘要，突出关键参与者、起因与后果。"

    # 新增：强化了“绝对不可原样摘抄”的指令
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
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
    ]

    print(f"\n正在请求 Gemini API 处理：【{title}】...")
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

    # 核心测试：打印最终的停止原因
    if resp.candidates:
        finish_reason = resp.candidates[0].finish_reason.name
        print(f"-> API 返回状态 (finish_reason): {finish_reason}")
        
        # 1. 明确打印出究竟消耗了多少个 Output Token
        if resp.usage_metadata:
            print(f"-> 实际消耗的输出 Token: {resp.usage_metadata.candidates_token_count}")
            
        # 2. 使用 repr() 打印最原始的返回内容（不使用 strip，让换行和空格现原形）
        print(f"-> 原始输出面貌: {repr(resp.candidates[0].content.parts[0].text)}")
        
        if finish_reason == "RECITATION":
            print("[诊断] 触发了版权/抄袭拦截！")
        elif finish_reason == "SAFETY":
            print("[诊断] 触发了安全审查拦截！")
        elif finish_reason == "MAX_TOKENS":
            print("[诊断] 确实达到了 Token 上限！请查看上面的原始输出是否有大量重复空白。")
        elif finish_reason == "STOP":
            print("[诊断] 正常生成完毕。")
            
    return resp.text.strip()

# ================= 测试执行区 =================
if __name__ == "__main__":
    # 直接使用你刚才截断的数据
    test_era = "拂晓时代" 
    test_title = "石源诸神的诞生"
    test_paragraphs = [
        "石源诸神被描述为“先于人类的司辰”“最初的司辰”，是最早一批统治世间的司辰。[1]目前看来石源诸神并不真的是从一块大石头上诞生的，虽然在进入蜘蛛之门时的描述提到“（丝滑沙地之下）是许久以前诸神由之诞生的石头”，但种种线索表明石源神是在诞生之后才进入漫宿的。[2]而且伤疤的戒律的呈递中也提到“漫宿是由石源众神构建的梦境堡垒”。为了解释这种矛盾，我们或许可以猜测石源诸神在修建漫宿的基础时，使用了来自他们诞生之地的材料。毕竟漫宿在最初的时候很可能和今天的月亮居屋一样是空无一物的。",
        "虽然关于拂晓时代早期，石源诸神依次诞生的资料几乎没有。但我们还是可以根据现有的一些线索进行合理推测：考虑到当时辉光尚未降临、漫宿无人知晓，就连虚界也尚未产生，石源诸神只有可能诞生在醒时或者林地。再考虑到石源诸神的权柄大多与自然现象相关（燧石与天空、浪潮与大海等等），笔者倾向于石源诸神是在醒时诞生的自然神。这也符合现实世界中各民族神话诞生与发展的一般规律，即从自然神开始，向人文神演进。",
        "在石源诸神都是自然神这个假设的基础上，我们甚至可以推测祂们的诞生顺序。其中最明晰的是逆孵之卵，他是作者钦定的起源神，对应着各民族神话中常见的宇宙卵模型。此类神话一般将世界的起源描述为一颗卵，卵壳破裂后流出的物质创造了世间万物。[3]其他几位石源神并没有明确的信息能够敲定祂们诞生的具体时间，但原文对祂们外形以及权柄的描述还是有所暗示。在这里对照现实世界的历史提出一种理论，仅供大家参考。",
        "当然，还是要再次强调一下，石源诸神的诞生时间与方式在本文撰成的当下仍是未知的。以上内容也只是结合现实世界的历史进行的推测，一切以游戏及其他同世界观作品的设定为准。"
    ]

    try:
        # 诞生事件不属于严重冲突
        result = summarise_event_text(test_era, test_title, test_paragraphs, False)
        
        print("\n" + "="*40)
        print("最终生成的摘要输出：")
        print(result)
        print("="*40)
    except Exception as e:
        print(f"\n[运行报错] {e}")