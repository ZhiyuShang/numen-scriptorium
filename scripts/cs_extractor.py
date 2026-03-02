import os
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import load_json_file, merge_data, save_json

# Paths
CS_ROOT = r"D:\SteamLibrary\steamapps\common\Cultist Simulator\cultistsimulator_Data\StreamingAssets\content"
CORE_ELEMENTS = os.path.join(CS_ROOT, "core", "elements")
LOC_ELEMENTS = os.path.join(CS_ROOT, "loc_zh-hans", "elements")
CORE_RECIPES = os.path.join(CS_ROOT, "core", "recipes")
LOC_RECIPES = os.path.join(CS_ROOT, "loc_zh-hans", "recipes")

OUTPUT_DIR = r"D:\Qwen3\data"

# Which element files to treat as items
ITEM_FILES = [
    "books_lore.json",
    "books_language.json",
    "books_other.json",
    "tools.json",
    "ingredients.json",  
    "fragments.json",
    "influences.json"
]

# Which recipe files contain vault explore info
VAULT_RECIPE_FILES = [
    "explore_vaults_a_capital.json",
    "explore_vaults_b_shires.json",
    "explore_vaults_c_continent.json",
    "explore_vaults_d_landbeyondforest.json",
    "explore_vaults_e_rendingmountains.json",
    "explore_vaults_f_loneandlevelsands.json",
    "explore_vaults_g_eveningisles.json",
    "explore_vaults_h_floating.json"
]


def infer_item_type_from_filename(filename: str) -> str:
    """
    Infer item type from element filename.
    - contains 'books' -> 'book'
    - contains 'tools' -> 'tool'
    - contains 'ingredient' -> 'ingredient'
    - otherwise -> 'item'
    """
    lower = filename.lower()
    if "books" in lower:
        return "book"
    if "tools" in lower:
        return "tool"
    if "ingredient" in lower:
        return "ingredient"
    if "fragment" in lower:
        return "fragment"
    if "influence" in lower:
        return "influence"
    return "item"

def load_items():
    """Load books/tools/ingredients from elements, and annotate _source_file on each element."""
    all_items = []
    for filename in ITEM_FILES:
        print(f"[Elements] Processing {filename}...")
        core_path = os.path.join(CORE_ELEMENTS, filename)
        loc_path = os.path.join(LOC_ELEMENTS, filename)

        if not os.path.exists(core_path):
            print(f"Warning: {core_path} not found.")
            continue

        core_data = load_json_file(core_path)
        loc_data = load_json_file(loc_path) if os.path.exists(loc_path) else None

        merged = merge_data(core_data, loc_data)
        for elem in merged:
            elem["_source_file"] = filename
        all_items.extend(merged)

    print(f"[Elements] Total items loaded: {len(all_items)}")
    return all_items


def load_book_readings():
    """
    Extract book reading content from study_1_books.json, used to add reading field to books.
    Non-books (tools/ingredients) generally do not appear here.
    """
    print("[Recipes] Loading book study recipes (study_1_books.json)...")
    recipes_core_path = os.path.join(CORE_RECIPES, "study_1_books.json")
    recipes_loc_path = os.path.join(LOC_RECIPES, "study_1_books.json")

    if not os.path.exists(recipes_core_path):
        print(f"Warning: {recipes_core_path} not found. No book reading content will be attached.")
        return {}

    recipes_core = load_json_file(recipes_core_path)
    recipes_loc = load_json_file(recipes_loc_path) if os.path.exists(recipes_loc_path) else None
    all_recipes = merge_data(recipes_core, recipes_loc)
    print(f"[Recipes] Loaded {len(all_recipes)} recipes from study_1_books.json.")

    item_recipe_map = {}
    for recipe in all_recipes:
        reqs = recipe.get("requirements", {})
        for req_id, _ in reqs.items():
            item_recipe_map[req_id] = recipe

    print(f"[Recipes] Built item_recipe_map for {len(item_recipe_map)} element ids (books).")
    return item_recipe_map


def extract_items_grouped(all_items, book_recipe_map):
    """
    Construct items grouped by type based on all_items + book_recipe_map:
    books / tools / ingredients / others
    """
    grouped = {
        "books": [],
        "tools": [],
        "ingredients": [],
        "fragments": [],
        "influences": [],
        "others": [],
    }

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
        }

        # If the element has a matching recipe in study_1_books, attach reading
        if item_id in book_recipe_map:
            recipe = book_recipe_map[item_id]
            item_data["reading"] = {
                "intro": recipe.get("startdescription", ""),
                "intro_cn": recipe.get("startdescription_cn", ""),
                "content": recipe.get("description", ""),
                "content_cn": recipe.get("description_cn", ""),
                "effects": recipe.get("effects", {}),
            }

        if item_type == "book":
            grouped["books"].append(item_data)
        elif item_type == "tool":
            grouped["tools"].append(item_data)
        elif item_type == "ingredient":
            grouped["ingredients"].append(item_data)
        elif item_type == "fragment":
            grouped["fragments"].append(item_data)
        elif item_type == "influence":
            grouped["influences"].append(item_data)
        else:
            grouped["others"].append(item_data)

    print(
        f"[Group] books={len(grouped['books'])}, "
        f"tools={len(grouped['tools'])}, "
        f"ingredients={len(grouped['ingredients'])}, "
        f"fragments={len(grouped['fragments'])}, "
        f"influences={len(grouped['influences'])}, "
        f"others={len(grouped['others'])}"
    )
    return grouped


# ---------- Vaults extraction ----------

def load_vault_recipes():
    """
    Extract vault-related recipes from multiple explore_vaults_*.json files.
    Returns list all_vault_recipes.
    """
    all_vault_recipes = []

    for filename in VAULT_RECIPE_FILES:
        core_path = os.path.join(CORE_RECIPES, filename)
        loc_path = os.path.join(LOC_RECIPES, filename)

        if not os.path.exists(core_path):
            print(f"Warning: vault recipe file {core_path} not found.")
            continue

        print(f"[Vaults] Loading vault recipes from {filename}...")
        core_data = load_json_file(core_path)
        loc_data = load_json_file(loc_path) if os.path.exists(loc_path) else None
        merged = merge_data(core_data, loc_data)

        # Some files might be {"recipes": [...]}, others are directly a list
        if isinstance(merged, dict) and "recipes" in merged:
            recs = merged["recipes"]
        elif isinstance(merged, list):
            recs = merged
        else:
            recs = []

        all_vault_recipes.extend(recs)

    print(f"[Vaults] Total vault recipes loaded: {len(all_vault_recipes)}")
    return all_vault_recipes


def extract_vaults(all_vault_recipes):
    """
    Extract configuration for each vault from vault explore recipes.
    Strategy:
    - Iterate through all recipes, find those with vault id in requirements (e.g. 'vaulteveningisles1')
    - Accumulate descriptions for different stages (setup / success etc.) using vault id as key
    """
    vaults = {}

    for recipe in all_vault_recipes:
        rid = recipe.get("id", "")
        label = recipe.get("label", "")
        label_cn = recipe.get("label_cn", label)  # If merge_data did localization merge, label_cn might already exist here
        reqs = recipe.get("requirements", {}) or {}
        aspects = recipe.get("aspects", {}) or {}

        # 1. Find vault id in requirements (simple rule: key starts with 'vault')
        vault_ids = [k for k in reqs.keys() if k.lower().startswith("vault")]
        if not vault_ids:
            continue  # explore recipe unrelated to vault

        for vid in vault_ids:
            r_id_lower = rid.lower()
            v = vaults.setdefault(
                    vid,
                    {
                        "id": vid,
                        "name": None,
                        "name_cn": None,
                        "aspects": {},
                        "type": "vault",
                        "descriptions": {
                            "setup_start": {"en": "", "cn": ""},
                            "success_start": {"en": "", "cn": ""},
                            "success_desc": {"en": "", "cn": ""},
                        },
                    },
                )

            # Merge aspects (if any)
            if aspects:
                # Simple merge: latter overwrites on value conflict
                v["aspects"].update(aspects)

            # Try to use the earliest seen label as vault name
            if not v["name"] and label:
                v["name"] = label
            if not v["name_cn"] and recipe.get("label_cn"):
                v["name_cn"] = recipe.get("label_cn")
            
            # Chinese text is usually in *_cn fields after merge_data, here we only grab English/Chinese descriptions
            start_en = recipe.get("startdescription", "")
            start_cn = recipe.get("startdescription_cn", "")
            desc_en = recipe.get("description", "")
            desc_cn = recipe.get("description_cn", "")

            if "setup" in r_id_lower:
                # Description before expedition starts
                if start_en:
                    v["descriptions"]["setup_start"]["en"] = start_en
                if start_cn:
                    v["descriptions"]["setup_start"]["cn"] = start_cn
            elif "success" in r_id_lower:
                # Description after expedition success
                if start_en:
                    v["descriptions"]["success_start"]["en"] = start_en
                if start_cn:
                    v["descriptions"]["success_start"]["cn"] = start_cn
                if desc_en:
                    v["descriptions"]["success_desc"]["en"] = desc_en
                if desc_cn:
                    v["descriptions"]["success_desc"]["cn"] = desc_cn
            else:
                # Other stages (if any), you can add fail / intermediate logic later
                pass

    print(f"[Vaults] Extracted {len(vaults)} vault entries.")
    return list(vaults.values())


def run():
    print("Starting Cultist Simulator Extraction (elements + vaults, grouped)...")

    # 1. Elements -> books/tools/ingredients/others
    all_items = load_items()
    book_recipe_map = load_book_readings()
    grouped_items = extract_items_grouped(all_items, book_recipe_map)

    # 2. Vault recipes -> vaults
    vault_recipes = load_vault_recipes()
    vaults = extract_vaults(vault_recipes)

    # 3. Save grouped JSON
    output_file = os.path.join(OUTPUT_DIR, "cs_raw_data.json")
    data_to_save = {
        "game": "Cultist Simulator",
        "books": grouped_items["books"],
        "tools": grouped_items["tools"],
        "ingredients": grouped_items["ingredients"],
        "fragments": grouped_items["fragments"],
        "influences": grouped_items["influences"],
        "others": grouped_items["others"],
        "vaults": vaults,
    }
    save_json(data_to_save, output_file)

    print(f"[Save] books={len(grouped_items['books'])}, "
          f"tools={len(grouped_items['tools'])}, "
          f"ingredients={len(grouped_items['ingredients'])}, "
          f"fragments={len(grouped_items['fragments'])}, "
          f"influences={len(grouped_items['influences'])}, "
          f"others={len(grouped_items['others'])}, "
          f"vaults={len(vaults)}")
    print(f"Saved to {output_file}")
    print("Done.")


if __name__ == "__main__":
    run()