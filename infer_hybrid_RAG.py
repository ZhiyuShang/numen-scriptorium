import os
import json
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import chromadb

# ================= 配置区 =================
BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_DIR = os.path.join("outputs", "qwen3_0_6b_boh_qlora")

# ================= 1. RAG 检索引擎 =================
class HybridRetriever:
    def __init__(self):
        print("[RAG] 正在初始化混合检索引擎...")
        
        # 1. 加载本地向量库 (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_data")
        self.collection = self.chroma_client.get_collection(name="mansus_lore")
        
        # 加载嵌入模型 (部署在 GPU 上加速检索)
        self.embedder = SentenceTransformer('moka-ai/m3e-base', device='cuda')
        
        # 2. 加载知识图谱三元组
        with open("kg_triplets.json", "r", encoding="utf-8") as f:
            self.triplets = json.load(f)
            
        # 3. 加载实体别名映射 (用于从用户问题中提取实体)
        with open("data\hours_merged.json", "r", encoding="utf-8") as f:
            hours_data = json.load(f)
            self.alias_map = {}
            for hour in hours_data.get("hours", []):
                standard_name = hour.get("name_cn", "")
                for alias in hour.get("aliases", []):
                    if alias.strip():
                        self.alias_map[alias.strip()] = standard_name

    def extract_entities(self, query: str):
        """用简单的字符串匹配从 Query 中提取提到的司辰实体"""
        found_entities = set()
        for alias, std_name in self.alias_map.items():
            if alias in query:
                found_entities.add(std_name)
        return list(found_entities)

    def retrieve(self, query: str, top_k: int = 3):
        print(f"\n[RAG] 正在分析问题: '{query}'")
        
        # --- 分支 A: 知识图谱检索 ---
        entities = self.extract_entities(query)
        kg_context = []
        if entities:
            print(f"[RAG] -> 图谱引擎: 捕捉到实体 {entities}")
            for t in self.triplets:
                if t["head_name"] in entities or t["tail_name"] in entities:
                    # 将三元组翻译成人类可读的句子
                    rel = t["relation"]
                    if rel == "HAS_ORIGIN":
                        kg_context.append(f"- {t['head_name']} 的起源是 {t['tail_name']}")
                    elif rel == "BELONGS_TO":
                        kg_context.append(f"- {t['head_name']} 属于派系 {t['tail_name']}")
                    elif rel == "PARTICIPATED_IN":
                        kg_context.append(f"- {t['head_name']} 参与了历史事件：{t['tail_name']}")
        
        # --- 分支 B: 向量库语义检索 ---
        print("[RAG] -> 向量引擎: 正在进行语义搜索...")
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        vector_context = results['documents'][0] if results['documents'] else []

        # --- 组装最终的上下文提示词 (Context) ---
        context_str = "【知识图谱确定的事实】\n"
        context_str += "\n".join(kg_context) if kg_context else "无直接图谱关联。"
        context_str += "\n\n【相关历史文献摘要】\n"
        context_str += "\n".join(vector_context)
        
        return context_str


# ================= 2. 语言模型引擎 =================
def load_llm():
    print("[LLM] 正在加载 Qwen3-0.6B 与 LoRA 权重...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, LORA_DIR)
    model.eval()
    return tokenizer, model

# def generate_answer(tokenizer, model, instruction: str, user_input: str, max_new_tokens: int = 256):
#     prompt = f"指令：{instruction}\n输入：\n{user_input}\n回答："
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=True,
#             temperature=0.6, 
#             top_p=0.9,
#             eos_token_id=tokenizer.eos_token_id,
#         )

#     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if "回答：" in text:
#         text = text.split("回答：", 1)[1]
#     return text.strip()

def generate_answer(tokenizer, model, instruction: str, rag_context: str, text_to_translate: str, max_new_tokens: int = 512):
    # 【核心修改 1：重构 Prompt 模板】
    # 将 RAG 内容严格限制在“参考资料”区域，并在最后再次强调翻译指令，防止模型遗忘。
    prompt = (
        f"指令：{instruction}\n\n"
        f"【专有名词与背景参考】（仅作翻译时的词汇对齐与语境参考，切勿直接摘抄或总结）：\n"
        f"------------------------\n"
        f"{rag_context}\n"
        f"------------------------\n\n"
        f"【需要翻译的英文原文】：\n{text_to_translate}\n\n"
        f"回答（中文翻译）："
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,      
            top_p=0.85,
            repetition_penalty=1.15, # 【防复读】
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 截取“回答（中文翻译）：”之后的内容
    if "回答（中文翻译）：" in text:
        text = text.split("回答（中文翻译）：", 1)[1]
    return text.strip()

# ================= 3. 主程序流水线 =================
if __name__ == "__main__":
    # 1. 启动引擎
    retriever = HybridRetriever()
    tokenizer, model = load_llm()

    # 2. 制定系统指令 (专为 0.6B 小模型 + RAG 优化)
    # 参考上下文，且保持文风
    system_instruction = (
    "你是一名翻译者，需要将英文文本翻译为中文。\n"
    "要求：\n"
    "1. 保留所有专有名词（包括人物、地名），不要将它们翻译为《司辰之书》；\n"
    "2. 在保证忠实的前提下，让整体语气更接近游戏《司辰之书》，适度使用象征与暗示。\n"
    "请翻译下面这段英文故事："
    )

    # 3. 开始对话测试
    queries = [
       "Name: The Sun's Design\nType: tablet\n"
       "Description: A scorched slab of black corundum, minutely scratched on every side with intricate ideoglyphs.\n "
       "In the city of Emesa, beneath the Church of the Holy Belt, in a sarcophagus of black corundum, Elagabalus lies: accursed of Janus, neither Long nor mortal, neither man nor woman, neither a liar nor a speaker of truth, neither real nor imagined. On his light-suffused skin is made manifest the Sun-in-Splendour's grand design...\n Elagabalus is the source of one-half of this text. The source of the other-half is obscure, but its power is evident. It is impossible to be certain if the Sun really planned for us to enter Eternity. It is impossible to be sure if the Grail, the Vagabond, and the Forge, stole this birthright from us, or saved us from it. But there is a great secret here."
    ]

    for q in queries:
        start_time = time.time()
        
        # 步骤 A：混合检索提取上下文
        context = retriever.retrieve(q, top_k=1)
        
        # 步骤 C：模型推理生成
        print("\n[LLM] 正在结合设定进行翻译...")
        answer = generate_answer(
            tokenizer=tokenizer, 
            model=model, 
            instruction=system_instruction, 
            rag_context=context,           # 独立的 RAG 上下文传入
            text_to_translate=q            # 纯净的英文原文传入
            )
        
        print("\n" + "="*50)
        print(f"提问: {q}")
        print("-" * 50)
        print(answer)
        print("="*50)
        print(f"耗时: {time.time() - start_time:.2f} 秒\n")