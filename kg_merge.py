import json
import os

def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"[错误] 找不到文件: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def build_knowledge_graph():
    print("[1] 正在加载数据...")
    hours_data = load_json("data/hours_merged.json")
    history_data = load_json("data/mansus_history_events_rag.json")

    triplets = []
    alias_map = {}

    print("[2] 正在解析司辰实体，提取内部关系 (起源、派系)...")

    hours_list = hours_data.get("hours", [])
    for hour in hours_list:
        hour_id = hour.get("id", "")
        hour_name = hour.get("name_cn", "")
        if not hour_id:
            continue
        
        # 提取起源 (HAS_ORIGIN)
        for origin in hour.get("origin", []):
            triplets.append({
                "head_id": hour_id, "head_name": hour_name,
                "relation": "HAS_ORIGIN",
                "tail_id": f"origin.{origin}", "tail_name": origin
            })
            
        # 提取派系 (BELONGS_TO)
        for faction in hour.get("factions", []):
            triplets.append({
                "head_id": hour_id, "head_name": hour_name,
                "relation": "BELONGS_TO",
                "tail_id": f"faction.{faction}", "tail_name": faction
            })
            
        # 构建倒排索引映射字典，用于后续在历史文本中“抓取”司辰
        for alias in hour.get("aliases", []):
            if alias.strip():
                # 记录别名对应的司辰 ID 和标准名称
                alias_map[alias.strip()] = {"id": hour_id, "name": hour_name}

    print(f" -> 提取了 {len(alias_map)} 个别名用于实体链接匹配。")

    print("[3] 正在扫描历史事件，建立事件参与关系 (PARTICIPATED_IN)...")
    # 遍历漫宿历史的每一个时代和事件
    for era_name, era_obj in history_data.items():
        events = era_obj.get("events", {})
        
        for event_title, event_obj in events.items():
            # 将主事件的段落和摘要拼成一段完整文本用于检索
            texts_to_search =  [event_obj.get("summary_cn", "")] #+ event_obj.get("paragraphs", []) 
            full_text = "\n".join(texts_to_search)
            
            # 使用别名映射表在文本中寻找司辰的踪迹
            matched_hours = set()
            for alias, hour_info in alias_map.items():
                if alias in full_text:
                    matched_hours.add((hour_info["id"], hour_info["name"]))
                    
            # 如果找到，则生成参与事件的三元组
            for h_id, h_name in matched_hours:
                triplets.append({
                    "head_id": h_id, "head_name": h_name,
                    "relation": "PARTICIPATED_IN",
                    "tail_id": f"event.{event_title}", "tail_name": event_title
                })
                
            # 同样地，扫描子事件 (h4)
            for sub_title, sub_obj in event_obj.get("subevents", {}).items():
                sub_texts = sub_obj.get("paragraphs", []) + [sub_obj.get("summary_cn", "")]
                sub_full_text = "\n".join(sub_texts)
                
                sub_matched = set()
                for alias, hour_info in alias_map.items():
                    if alias in sub_full_text:
                        sub_matched.add((hour_info["id"], hour_info["name"]))
                        
                for h_id, h_name in sub_matched:
                    triplets.append({
                        "head_id": h_id, "head_name": h_name,
                        "relation": "PARTICIPATED_IN",
                        "tail_id": f"event.{sub_title}", "tail_name": sub_title
                    })

    print(f"[4] 构建完成！共生成 {len(triplets)} 条知识图谱三元组边。")
    
    output_file = "kg_triplets.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
    print(f"[5] 数据已保存至 {output_file}")

if __name__ == "__main__":
    build_knowledge_graph()