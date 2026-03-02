import json
from collections import Counter, defaultdict

# 替换为你的实际数据文件路径，例如 "teammate_format_corrected.json" 或 "data/boh_raw_data_items.json"
DATA_FILE = "data/boh_raw_data_items.json" 

def analyze_aspected_items():
    print(f"正在读取数据文件: {DATA_FILE} ...\n")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {DATA_FILE}，请检查路径。")
        return

    # 兼容你的 teammate_format (字典里有 "items" 列表) 
    # 或者直接就是一个大列表
    items = data.get("items", []) if isinstance(data, dict) else data

    prefix_counter = Counter()
    # 用来记录每个前缀对应的中文名样例，方便你做决策
    prefix_examples = defaultdict(set) 

    for item in items:
        item_type = (item.get("type") or "").strip().lower()
        if item_type == "aspecteditem":
            item_id = (item.get("id") or "").strip().lower()
            
            if "." in item_id:
                # 提取点号前面的部分作为前缀
                prefix = item_id.split(".", 1)[0]
                prefix_counter[prefix] += 1
                
                # 收集中文名样例（最多收集3个用于展示）
                name_cn = item.get("name_cn", "无中文名")
                if len(prefix_examples[prefix]) < 3:
                    prefix_examples[prefix].add(name_cn)
            else:
                # 记录没有 '.' 的异常 id
                prefix_counter["[NO_DOT]"] += 1
                prefix_examples["[NO_DOT]"].add(item.get("name_cn", item_id))

    # 按出现频率从高到低排序
    sorted_prefixes = prefix_counter.most_common()

    print("=" * 60)
    print(f"📊 AspectedItem ID 前缀频率统计 (共找到 {len(sorted_prefixes)} 种前缀)")
    print("=" * 60)
    print(f"{'前缀 (Prefix)':<15} | {'数量':<5} | {'中文名样例'}")
    print("-" * 60)

    for prefix, count in sorted_prefixes:
        # 将 set 转为逗号分隔的字符串
        examples_str = ", ".join(list(prefix_examples[prefix]))
        print(f"{prefix:<15} | {count:<5} | {examples_str}")

    print("-" * 60)
    print("💡 建议：")
    print("1. 挑选【数量多】且【类别明确】的前缀（如 dog, bed 等），加入到你的 TYPE_MAP_ZH 字典中。")
    print("2. 数量极少（如 1~2个）或难以统一定义的（如 jerry, pitcher），直接无视，让它们安全回退为'物品'。")

if __name__ == "__main__":
    analyze_aspected_items()