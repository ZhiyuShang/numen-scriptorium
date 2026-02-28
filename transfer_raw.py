"""
《密教模拟器》(Cultist Simulator) 原始实体数据集 (cs_raw_data.json)
包含游戏内各类物品、书籍文献、探索地点等核心实体的中英双语文本与属性数据。
该结构专为提取游戏文本、构建知识图谱与 RAG 语料库设计。

==================== 根节点结构 ====================
{
    "game"        : str,          # 游戏标识名称，固定为 "Cultist Simulator"
    "books"       : List[Dict],   # 书籍/文献实体列表 (包含阅读文本与阅读产出)
    "tools"       : List[Dict],   # 工具/器物实体列表
    "ingredients" : List[Dict],   # 原料/配剂实体列表
    "fragments"   : List[Dict],   # 密传/知识碎片实体列表
    "influences"  : List[Dict],   # 影响/短暂状态实体列表
    "others"      : List[Dict],   # 其他未分类实体列表
    "vaults"      : List[Dict]    # 藏宝地/探险地点实体列表
}

==================== 实体结构详情 ====================

1. 基础物品实体 (适用于 tools, ingredients, fragments, influences, others)
记录游戏内卡牌的基础属性与中英双语描述。
- id              (str) : 唯一标识符（例："dehoris1"）
- name            (str) : 英文名称
- name_cn         (str) : 中文译名
- description     (str) : 英文背景描述
- description_cn  (str) : 中文背景描述
- aspects         (Dict[str, int]) : 性相字典，记录物品拥有的性相及等级（例：{"text": 1}）
- type            (str) : 实体类型标识（例："tool", "ingredient"）

2. 书籍文献实体 (books)
在继承基础物品实体的所有字段外，包含独有的阅读交互数据。
- [继承基础实体所有字段...]
- reading         (Dict): 阅读交互数据字典
  |- intro        (str) : 开始阅读时的英文引言提示
  |- intro_cn     (str) : 开始阅读时的中文引言提示
  |- content      (str) : 阅读完成后的英文正文/感悟
  |- content_cn   (str) : 阅读完成后的中文正文/感悟
  |- effects      (Dict[str, int]) : 阅读完成后的物品变化字典（例：{"dehoris1": -1, "erudition": 1}，负数代表消耗，正数代表产出）

3. 探险地点实体 (vaults)
记录游戏内的藏宝地及探险各阶段的剧情文本。
- id              (str) : 唯一标识符（例："vaultcapital1"）
- name            (str) : 英文地点名称
- name_cn         (str) : 中文地点名称
- aspects         (Dict[str, int]) : 地点性相（通常为空或包含探险阻碍性相）
- type            (str) : 实体类型标识（固定为 "vault"）
- descriptions    (Dict): 探险各阶段的双语剧情文本字典
  |- setup_start  (Dict[str, str]) : 探险准备/遭遇阻碍时的文本 (含 "en" 和 "cn" 键)
  |- success_start(Dict[str, str]) : 突破阻碍/进入密室时的文本 (含 "en" 和 "cn" 键)
  |- success_desc (Dict[str, str]) : 探险成功/结算战利品时的文本 (含 "en" 和 "cn" 键)
"""

import json

def convert_to_teammate_format(input_filepath, output_filepath):
    # 1. 读取原始的多层级数据
    with open(input_filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # 2. 初始化队友期望的单列表结构
    teammate_format = {"items": []}
    
    # 定义需要被压平的实体分类
    categories = ["books", "tools", "ingredients", "fragments", "influences", "others", "vaults"]
    
    # 3. 遍历所有分类，将物品塞入同一个 items 列表
    for category in categories:
        if category not in raw_data:
            continue
            
        for raw_item in raw_data[category]:
            # 提取所有实体的共性基础属性
            item = {
                "id": raw_item.get("id"),
                "name": raw_item.get("name"),
                "name_cn": raw_item.get("name_cn"),
                "description": raw_item.get("description", ""),
                "description_cn": raw_item.get("description_cn", ""),
                "aspects": raw_item.get("aspects", {}),
                "type": raw_item.get("type")
            }
            
            # 如果是地点(vaults)，保留其特有的 descriptions
            if "descriptions" in raw_item:
                item["descriptions"] = raw_item["descriptions"]

            # 4. 重点处理书籍(books)特有的 reading 嵌套结构
            if "reading" in raw_item:
                reading_data = raw_item["reading"]
                
                # 组装英文 readings 数组
                if "intro" in reading_data or "content" in reading_data:
                    item["readings"] = [{
                        "intro": reading_data.get("intro", ""),
                        "content": reading_data.get("content", "")
                    }]
                    
                # 组装中文 readings_cn 数组
                if "intro_cn" in reading_data or "content_cn" in reading_data:
                    item["readings_cn"] = [{
                        "intro": reading_data.get("intro_cn", ""),
                        "content": reading_data.get("content_cn", "")
                    }]
                    
                # 加回丢失的 effects 字典（放在同级目录）
                if "effects" in reading_data:
                    item["effects"] = reading_data["effects"]

            # 将组装好的条目推入大列表
            teammate_format["items"].append(item)
            
    # 5. 输出为新的 JSON 文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(teammate_format, f, ensure_ascii=False, indent=2)

    print(f"[转换成功] 共处理了 {len(teammate_format['items'])} 个条目，已保存至 {output_filepath}")
    print("-> 已剔除 'results' 字段。")
    print("-> 已将原 'reading' 中的 'effects' 独立提取。")

if __name__ == "__main__":
    # 确保 cs_raw_data.json 和脚本在同一目录下
    convert_to_teammate_format("data\cs_raw_data.json", "data\corrected_cs_raw_items.json")