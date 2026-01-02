import os
import json
from common import load_json_file, merge_data, save_json

# Paths
CS_ROOT = r"D:\SteamLibrary\steamapps\common\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content"
CORE_ELEMENTS = os.path.join(CS_ROOT, "core", "elements")
LOC_ELEMENTS = os.path.join(CS_ROOT, "loc_zh-hans", "elements")
CORE_RECIPES = os.path.join(CS_ROOT, "core", "recipes")
LOC_RECIPES = os.path.join(CS_ROOT, "loc_zh-hans", "recipes")

OUTPUT_DIR = r"D:\Qwen3\data"

# Configuration: Which files to process for items
ITEM_FILES = [
    "books_lore.json",
    "books_language.json",
    "books_other.json",
    "tools.json",
    "ingredients.json",  
]

def infer_item_type_from_filename(filename: str) -> str:
    """
    根据元素文件名推断物品类型。
    - 包含 'books' -> 'book'
    - 包含 'tools' -> 'tool'
    - 包含 'ingredient' -> 'ingredient'
    - 否则 -> 'item'
    """
    lower = filename.lower()
    if "books" in lower:
        return "book"
    if "tools" in lower:
        return "tool"
    if "ingredient" in lower:
        return "ingredient"
    return "item"

def run():
    print("Starting Cultist Simulator Extraction (generalized)...")

    # 1. Load Items from multiple files (books / tools / ingredients /...)
    all_items = []
    for filename in ITEM_FILES:
        print(f"Processing {filename}...")
        core_path = os.path.join(CORE_ELEMENTS, filename)
        loc_path = os.path.join(LOC_ELEMENTS, filename)
        
        if not os.path.exists(core_path):
            print(f"Warning: {core_path} not found.")
            continue
            
        core_data = load_json_file(core_path)
        loc_data = load_json_file(loc_path) if os.path.exists(loc_path) else None
        
        merged = merge_data(core_data, loc_data)
        # 在每个元素上标记来源文件，方便后面推断 type
        for elem in merged:
            elem["_source_file"] = filename
        all_items.extend(merged)
        
    print(f"Total items loaded from elements: {len(all_items)}")

    # 2. Load Recipes (only study_1_books for now, since tools/ingredients usually have no 'reading')
    print("Loading Recipes (study_1_books)...")
    recipes_core_path = os.path.join(CORE_RECIPES, "study_1_books.json")
    recipes_loc_path = os.path.join(LOC_RECIPES, "study_1_books.json")

    if os.path.exists(recipes_core_path):
        recipes_core = load_json_file(recipes_core_path)
        recipes_loc = load_json_file(recipes_loc_path) if os.path.exists(recipes_loc_path) else None
        all_recipes = merge_data(recipes_core, recipes_loc)
        print(f"Loaded {len(all_recipes)} recipes from study_1_books.json.")
    else:
        all_recipes = []
        print(f"Warning: {recipes_core_path} not found. No reading content will be attached.")

    # 3. Map Recipes to Items (by requirement id)
    item_recipe_map = {}
    for recipe in all_recipes:
        reqs = recipe.get("requirements", {})
        for req_id, req_val in reqs.items():
            # 如果多个 recipe 要求同一个元素，这里会被后一个覆盖；
            # 如果你以后需要多个阅读版本，可以改成列表收集。
            item_recipe_map[req_id] = recipe

    print(f"Built item_recipe_map for {len(item_recipe_map)} element ids.")

    # 4. Build Final Data
    extracted_items = []
    
    for item in all_items:
        item_id = item.get("id")
        if not item_id:
            continue

        source_file = item.get("_source_file", "")
        item_type = infer_item_type_from_filename(source_file)

        item_data = {
            "id": item_id,
            "name": item.get("label", "Unknown"),
            "name_cn": item.get("label_cn", "Unknown"),
            "description": item.get("description", ""),
            "description_cn": item.get("description_cn", ""),
            "aspects": item.get("aspects", {}),
            "type": item_type,
            # "icon": item.get("icon", item_id),  # 如果以后需要图标，可以打开
        }

        # 只要该元素在 study_1_books 里有配套 recipe，就为它附加 'reading'
        # 对于 tools / ingredients，通常不会有；不会强行赋值。
        if item_id in item_recipe_map:
            recipe = item_recipe_map[item_id]
            item_data["reading"] = {
                "intro": recipe.get("startdescription", ""),
                "intro_cn": recipe.get("startdescription_cn", ""),
                "content": recipe.get("description", ""),
                "content_cn": recipe.get("description_cn", ""),
                "effects": recipe.get("effects", {}),
            }
        
        extracted_items.append(item_data)

    # 5. Save
    output_file = os.path.join(OUTPUT_DIR, "cs_raw_data.json")
    save_json({"game": "Cultist Simulator", "items": extracted_items}, output_file)
    print(f"Extracted {len(extracted_items)} items. Saved to {output_file}")
    print("Done.")

if __name__ == "__main__":
    run()