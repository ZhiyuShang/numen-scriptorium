import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"[错误] 找不到文件: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vector_db():
    # 推荐使用 m3e-base，对中文文本的检索效果非常好，且体积小
    print("[1] 正在加载嵌入模型...")
    embedder = SentenceTransformer('moka-ai/m3e-base', device='cuda')

    print("[2] 初始化本地 Chroma 向量数据库...")
    # 这会在当前目录下创建一个名为 "chroma_data" 的文件夹来持久化存储数据
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    
    # 创建或获取一个集合（Collection），相当于关系型数据库里的"表"
    collection = chroma_client.get_or_create_collection(name="mansus_lore")

    print("[3] 正在读取 JSON 数据...")
    hours_data = load_json("data/hours_merged.json")
    history_data = load_json("data/mansus_history_events_rag.json")

    documents = []   # 存储纯文本块
    metadatas = []   # 存储元数据（用于过滤和与图谱联动）
    ids = []         # 存储唯一 ID

    print("[4] 正在处理司辰 (Hours) 文本...")
    for hour in hours_data.get("hours", []):
        hour_id = hour.get("id", "")
        desc = hour.get("desc_cn", "")
        name = hour.get("name_cn", "")
        
        if not hour_id or not desc:
            continue
            
        documents.append(f"【司辰档案】{name}：{desc}")
        metadatas.append({
            "type": "hour",
            "entity_id": hour_id,
            "entity_name": name
        })
        ids.append(f"doc_{hour_id}")

    print("[5] 正在处理漫宿历史事件 (History Events) 文本...")
    for era_name, era_obj in history_data.items():
        for event_title, event_obj in era_obj.get("events", {}).items():
            # 优先使用我们之前用大模型生成的精炼摘要
            summary = event_obj.get("summary_cn", "")
            if not summary:
                # 如果没有摘要，就把原段落拼起来
                summary = "\n".join(event_obj.get("paragraphs", []))
                
            if summary.strip():
                documents.append(f"【历史事件】{era_name} - {event_title}：\n{summary}")
                metadatas.append({
                    "type": "event",
                    "era": era_name,
                    "event_title": event_title
                })
                ids.append(f"doc_event_{event_title}")
                
            # 处理子事件 (h4)
            for sub_title, sub_obj in event_obj.get("subevents", {}).items():
                sub_summary = sub_obj.get("summary_cn", "")
                if not sub_summary:
                    sub_summary = "\n".join(sub_obj.get("paragraphs", []))
                
                if sub_summary.strip():
                    documents.append(f"【历史事件】{era_name} - {event_title} ({sub_title})：\n{sub_summary}")
                    metadatas.append({
                        "type": "subevent",
                        "era": era_name,
                        "parent_event": event_title,
                        "event_title": sub_title
                    })
                    ids.append(f"doc_subevent_{sub_title}")

    print(f"[6] 开始对 {len(documents)} 个文本块进行向量化并存入数据库 ...")
    # 批量进行向量化
    embeddings = embedder.encode(documents, show_progress_bar=True).tolist()

    # 批量存入 ChromaDB
    # 注意：如果数据量上万，建议分批次（Batch）存入。这里数据量在几百条左右，可以直接一次性插入。
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print("[7] 向量库构建完成！数据已持久化保存在 ./chroma_data 目录。")

if __name__ == "__main__":
    build_vector_db()