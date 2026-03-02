import json
import re


def _clean_json_like_text(text: str) -> str:
    text = re.sub(r"(?m)//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(?m)^\s*#.*$", "", text)
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def load_json_file(filepath):
    encodings = ["utf-8-sig", "utf-16", "utf-8", "latin-1"]
    last_error = None
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                try:
                    return json.load(f, strict=False)
                except json.JSONDecodeError as e:
                    f.seek(0)
                    cleaned = _clean_json_like_text(f.read())
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError as e2:
                        print(f"JSON Error in {filepath} with {enc}: {e2}")
                        last_error = e2
                        continue
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error loading {filepath} with {enc}: {e}")
            last_error = e
            continue
    print(f"Failed to load {filepath} with any encoding. Last error: {last_error}")
    return None


def get_id(item):
    if "id" in item:
        return item["id"]
    if "ID" in item:
        return item["ID"]
    return None


def merge_data(core_data, loc_data, filename=None):
    core_items = []
    if isinstance(core_data, dict):
        for _, v in core_data.items():
            if isinstance(v, list):
                core_items.extend(v)
    elif isinstance(core_data, list):
        core_items = core_data

    loc_map = {}
    if loc_data:
        loc_items = []
        if isinstance(loc_data, dict):
            for _, v in loc_data.items():
                if isinstance(v, list):
                    loc_items.extend(v)
        elif isinstance(loc_data, list):
            loc_items = loc_data
        for item in loc_items:
            uid = get_id(item)
            if uid:
                loc_map[uid] = item

    merged = []
    for item in core_items:
        uid = get_id(item)
        if uid and uid in loc_map:
            loc_item = loc_map[uid]
            for k, v in loc_item.items():
                if k in [
                    "label",
                    "Label",
                    "description",
                    "desc",
                    "Description",
                    "Desc",
                    "startdescription",
                    "slots",
                    "xexts",
                ]:
                    item[f"{k}_cn"] = v
        merged.append(item)
    return merged


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filepath}")
