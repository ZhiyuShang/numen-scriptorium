import json
import os
import sys
import re

def _clean_json_like_text(text: str) -> str:
    """
    Attempt to clean "JSON-like" text into a string accepted by json.loads:
    - Remove // comment lines
    - Remove /*... */ block comments
    - Remove leading # comments
    - Simply remove some obvious trailing commas (conservative approach)
    Note: This is a heuristic method, not guaranteed to handle all cases, but usually sufficient for CS files.
    """

    # 1) Remove // trailing comments
    #    Be careful not to delete http:// etc., here use a simple check " // followed by space or text and inline"
    text = re.sub(r"(?m)//[^\n]*", "", text)

    # 2) Remove /*... */ block comments (multi-line)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)

    # 3) Remove leading # comments
    text = re.sub(r"(?m)^\s*#.*$", "", text)

    # 4) Attempt to remove trailing commas:
    #    - In objects: ... , }  ->... }
    #    - In arrays: ... , ]  ->... ]
    text = re.sub(r",\s*([\]}])", r"\1", text)

    return text


def load_json_file(filepath):
    """
    Attempt to load JSON with multiple encodings:
    1. Direct json.load
    2. If failed, read as string -> clean -> json.loads
    """
    encodings = ['utf-8-sig', 'utf-16', 'utf-8', 'latin-1']
    last_error = None

    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                # Try strict parsing first
                try:
                    return json.load(f, strict=False)
                except json.JSONDecodeError as e:
                    # Fallback to "loose mode": read full text, clean then json.loads
                    f.seek(0)
                    raw_text = f.read()
                    cleaned = _clean_json_like_text(raw_text)
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

# def load_json_file(filepath):
#     encodings = ['utf-8-sig', 'utf-16', 'utf-8', 'latin-1']
#     for enc in encodings:
#         try:
#             with open(filepath, 'r', encoding=enc) as f:
#                 return json.load(f, strict=False)
#         except UnicodeDecodeError:
#             continue
#         except json.JSONDecodeError as e:
#             print(f"JSON Error in {filepath} with {enc}: {e}")
#             # Try to clean up JSON?
#             continue
#         except Exception as e:
#             print(f"Error loading {filepath} with {enc}: {e}")
#             return None
    
#     print(f"Failed to load {filepath} with any encoding.")
#     return None

def get_id(item):
    """Returns the value of 'id' or 'ID' from the item."""
    if 'id' in item:
        return item['id']
    if 'ID' in item:
        return item['ID']
    return None

def merge_data(core_data, loc_data, filename=None):
    """
    Merges localization data into core data.
    Matches items by 'id' or 'ID'.
    """
    # Normalize to list of items
    core_items = []
    if isinstance(core_data, dict):
        for k, v in core_data.items():
            if isinstance(v, list):
                core_items.extend(v)
    elif isinstance(core_data, list):
        core_items = core_data

    loc_map = {}
    if loc_data:
        loc_items = []
        if isinstance(loc_data, dict):
            for k, v in loc_data.items():
                if isinstance(v, list):
                    loc_items.extend(v)
        elif isinstance(loc_data, list):
            loc_items = loc_data
        
        for item in loc_items:
            uid = get_id(item)
            if uid:
                loc_map[uid] = item

    # Merge
    merged = []
    for item in core_items:
        uid = get_id(item)
        if uid and uid in loc_map:
            # Update with localized fields
            loc_item = loc_map[uid]
            for k, v in loc_item.items():
                # Update text fields and xexts (for BoH)
                if k in ['label', 'Label', 'description', 'desc','Description', 'Desc', 'startdescription', 'slots', 'xexts']: 
                    item[f'{k}_cn'] = v
                

        merged.append(item)

       
    
    return merged

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filepath}")