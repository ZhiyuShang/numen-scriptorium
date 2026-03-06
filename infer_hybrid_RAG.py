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

    def retrieve(self, query: str, top_k: int = 1):
        print(f"\n[RAG] 正在启动双引擎检索...")
        
        # --- 分支 A: 知识图谱提取 (精确词典) ---
        rag_dict = {}
        for alias, std_name in self.alias_map.items():
            if alias.lower() in query.lower() and len(alias) > 2:
                rag_dict[alias] = std_name
        print(f"[RAG] -> 图谱引擎命中术语: {rag_dict}")
                
        # --- 分支 B: 向量库语义检索 (背景故事) ---
        vector_context = ""
        try:
            query_embedding = self.embedder.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k
            )
            if results and results['documents'] and results['documents'][0]:
                # 将检索到的 top_k 个文段拼接起来
                vector_context = "\n".join(results['documents'][0])
                print(f"[RAG] -> 向量引擎命中背景: 获取了 {len(results['documents'][0])} 段相关设定。")
        except Exception as e:
            print(f"[RAG] -> 向量检索跳过或失败: {e}")

        return rag_dict, vector_context

    def retrieve_dict(self, query: str) -> dict:
        """纯粹的图谱词典提取器，返回字典供打小抄使用"""
        print(f"\n[RAG] 正在从知识图谱中扫描匹配的专有名词...")
        rag_dict = {}
        # 遍历我们在 hours_merge.json 里建立的别名映射表
        for alias, std_name in self.alias_map.items():
            # 大小写不敏感的字符串匹配
            if alias.lower() in query.lower():
                # 只有当英文原文确实比较长（大于2个字母，防止误匹配）才加入字典
                if len(alias) > 2: 
                    rag_dict[alias] = std_name
                    
        print(f"[RAG] -> 提取到专用词典: {rag_dict}")
        return rag_dict

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

def generate_answer(tokenizer, model, instruction: str, rag_dict: dict, vector_context: str, text_to_translate: str, max_new_tokens: int = 512):
    
    # 【外挂技巧 1：术语硬注入】（保留这个优秀机制）
    injected_text = text_to_translate
    for eng_term, cn_term in rag_dict.items():
        if eng_term in injected_text:
            injected_text = injected_text.replace(eng_term, f"{eng_term}({cn_term})")

    # 【外挂技巧 2：动态构建 System Prompt 提供背景】
    dynamic_system_prompt = instruction
    if vector_context:
        dynamic_system_prompt += (
            "\n\n【背景语境参考】\n"
            "以下内容为从《司辰之书》世界观数据库中检索到的相关背景故事。\n"
            "请仅将此作为理解上下文、隐喻和设定语气的参考，**绝对不要**直接翻译或抄袭参考文本：\n"
            f"\"\"\"{vector_context}\"\"\""
        )

    # 构建标准的消息格式
    messages = [
        {"role": "system", "content": dynamic_system_prompt},
        {"role": "user", "content": f"请翻译以下文本（括号内为强制规定的中文译名，请在最终翻译中直接使用该译名，去掉括号和英文）：\n{injected_text}"}
    ]
    
    # 转换为 Qwen2.5 专属格式
    text_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,       
            top_p=0.85,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id, 
        )

    # 截取新生成的部分
    input_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
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
        #context = retriever.retrieve(q, top_k=1)
        rag_dict, vector_context = retriever.retrieve(q, top_k=1) # 可以尝试把 top_k 改成 2 
        
        # 步骤 C：模型推理生成
        print("\n[LLM] 正在结合设定进行翻译...")
        answer = generate_answer(
            tokenizer=tokenizer, 
            model=model, 
            instruction=system_instruction, 
            rag_dict=rag_dict,             # 图谱引擎：强制术语
            vector_context=vector_context, # 向量引擎：背景故事
            text_to_translate=q            # 原文
        )
        
        print("\n" + "="*50)
        print(f"提问: {q}")
        print("-" * 50)
        print(answer)
        print("="*50)
        print(f"耗时: {time.time() - start_time:.2f} 秒\n")