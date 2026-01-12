import os
import json
from common import load_json_file, merge_data, save_json

# Paths
BOH_ROOT = r"D:\SteamLibrary\steamapps\common\Book of Hours\bh_Data\StreamingAssets\bhcontent"
CORE_ELEMENTS = os.path.join(BOH_ROOT, "core", "elements")
LOC_ELEMENTS = os.path.join(BOH_ROOT, "loc_zh-hans", "elements")

OUTPUT_DIR = r"D:\Qwen3\data"

# Configuration: Which files to process for items
ITEM_FILES = [
    "tomes.json",
    "journal.json",
    "correspondence_elements.json",
    "aspecteditems.json",
    "skills.json",
    "skills_r.json",
    "incidents_n.json",
    "abilities.json",
    "visitors.json",
    "visitors_embarking.json",      # extra visitors info (xexts)
    "_visitactedaspects.json",      # extra incident reactions
    "incidents_weather.json",       # weather entries
]


def infer_item_type_from_filename(filename: str) -> str:
    """
    Infer a semantic type from the element file name.
    Simple heuristic; adjust as needed.
    """
    lower = filename.lower()
    if "tomes" in lower or "tome" in lower:
        return "tome"
    if "journal" in lower:
        return "journal"
    if "correspondence" in lower:
        return "correspondence"
    if "aspecteditems" in lower:
        return "aspecteditem"
    if lower == "skills.json" or "_skills" in lower:
        return "skills"
    if "skills_r" in lower:
        return "skills_r"
    if "incidents_weather" in lower:
        return "weather"
    if "incident" in lower:
        return "incident"
    if "abilities" in lower:
        return "ability"
    if "visitors_embarking" in lower or "visitors" in lower:
        return "visitor"
    return "item"


def get_desc_fields(item: dict, is_cn: bool = False) -> str:
    """
    Robustly get description / desc / Desc / Desc_cn / desc_cn / description_cn.
    """
    if is_cn:
        return (
            item.get("Desc_cn")
            or item.get("desc_cn")
            or item.get("description_cn")
            or ""
        )
    else:
        return (
            item.get("Desc")
            or item.get("desc")
            or item.get("description")
            or ""
        )


def load_all_elements_for_lookup():
    """
    Loads ALL elements from the game to build a ID -> Name lookup map.
    Used for resolving IDs in xtriggers (e.g. 'numen.conf' -> 'Numen: Inescapable Confinement')
    """
    print("Building global lookup map...")
    lookup = {}

    for filename in os.listdir(CORE_ELEMENTS):
        if not filename.endswith(".json"):
            continue

        core_path = os.path.join(CORE_ELEMENTS, filename)
        loc_path = os.path.join(LOC_ELEMENTS, filename)

        core_data = load_json_file(core_path)
        if core_data is None:
            continue

        loc_data = load_json_file(loc_path) if os.path.exists(loc_path) else None

        merged = merge_data(core_data, loc_data)

        for item in merged:
            uid = item.get("id") or item.get("ID")
            if not uid:
                continue

            label = item.get("Label") or item.get("label")
            label_cn = item.get("Label_cn") or item.get("label_cn")

            if label:
                lookup[uid] = [label, label_cn]

    print(f"Lookup map built with {len(lookup)} entries.")
    return lookup


def parse_xexts_readings_only(xexts):
    """
    Original reading parser: parse xexts into structured reading list.
    ONLY handles keys like 'reading.lantern' or 'reading.lantern.intro'.
    Non-reading keys (scrutiny, contamination.*, befriend.*...) are ignored here
    and can be handled separately in future.
    """
    if not xexts:
        return None

    readings = {}

    for key, text in xexts.items():
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "reading":
            aspect = parts[1]  # e.g., 'lantern'
            if aspect not in readings:
                readings[aspect] = {"aspect": aspect}
            if len(parts) == 3 and parts[2] == "intro":
                readings[aspect]["intro"] = text
            elif len(parts) == 2:
                readings[aspect]["content"] = text

    return list(readings.values()) if readings else None


def parse_xtriggers(xtriggers, lookup_map):
    """
    Parses xtriggers to find results of reading/mastering.
    """
    if not xtriggers:
        return None

    results = []

    for trigger_key, effects in xtriggers.items():
        action_type = "unknown"
        aspect = "unknown"

        parts = trigger_key.split(".")
        if len(parts) >= 2:
            action_type = parts[0]  # reading, mastering
            aspect = parts[1]

        if action_type not in ["reading", "mastering"]:
            continue

        for effect in effects:
            effect_id = effect.get("id")
            if not effect_id:
                continue

            lookup_val = lookup_map.get(effect_id)
            if isinstance(lookup_val, list) and len(lookup_val) == 2:
                result_name, result_name_cn = lookup_val
            else:
                result_name = effect_id
                result_name_cn = effect_id

            results.append(
                {
                    "action": action_type,
                    "aspect": aspect,
                    "result_id": effect_id,
                    "result_name": result_name,
                    "result_name_cn": result_name_cn,
                    "level": effect.get("level", 1),
                    "type": "spawn" if effect.get("morpheffect") == "spawn" else "transform",
                }
            )

    return results if results else None


def run():
    print("Starting Book of Hours Extraction (grouped, with visitors & weather)...")

    # 1. Build Lookup Map
    lookup_map = load_all_elements_for_lookup()

    # 2. Load Target Items
    all_items = []
    for filename in ITEM_FILES:
        print(f"[Elements] Processing {filename}...")
        core_path = os.path.join(CORE_ELEMENTS, filename)
        loc_path = os.path.join(LOC_ELEMENTS, filename)

        core_data = load_json_file(core_path)
        if core_data is None:
            print(f"Warning: failed to load core {core_path}")
            continue

        loc_data = load_json_file(loc_path) if os.path.exists(loc_path) else None
        
        merged = merge_data(core_data, loc_data,filename)
        for elem in merged:
            elem["_source_file"] = filename
        all_items.extend(merged)

    print(f"[Elements] Total items loaded: {len(all_items)}")

    # 3. First pass: basic grouping by type
    grouped = {
        "tomes": [],
        "journals": [],
        "aspecteditems": [],
        "skills": [],
        "skills_r": [],
        "incidents": [],
        "abilities": [],
        "visitors": [],
        "weather": [],
        "others": [],
        # correspondence is loaded but will be ignored as per your note
    }

    # Temporary buckets for special merges
    visitors_base = []
    visitors_embarking = []
    incidents_base = []
    visitacted_entries = []
    weather_entries = []

    for item in all_items:
        item_id = item.get("ID") or item.get("id")
        if not item_id:
            continue

        source_file = item.get("_source_file", "")
        inferred_type = infer_item_type_from_filename(source_file)

        # Skip correspondence group in final output
        if inferred_type == "correspondence":
            continue

        # classify special-case files for later merge
        if "visitors_embarking" in source_file.lower():
            visitors_embarking.append(item)
            continue
        if "_visitactedaspects" in source_file.lower():
            visitacted_entries.append(item)
            continue
        if "incidents_weather" in source_file.lower():
            weather_entries.append(item)
            continue

        # normal items
        if inferred_type == "tome":
            group_key = "tomes"
        elif inferred_type == "journal":
            group_key = "journals"
        elif inferred_type == "aspecteditem":
            group_key = "aspecteditems"
        elif inferred_type == "skills":
            group_key = "skills"
        elif inferred_type == "skills_r":
            group_key = "skills_r"
        elif inferred_type == "incident":
            group_key = "incidents"
        elif inferred_type == "ability":
            group_key = "abilities"
        elif inferred_type == "visitor":
            group_key = "visitors"
        elif inferred_type == "weather":
            group_key = "weather"
        else:
            group_key = "others"

        item_data = {
            "id": item_id,
            "name": item.get("Label", item.get("label", "Unknown")),
            "name_cn": item.get("Label_cn", item.get("label_cn", "Unknown")),
            "description": get_desc_fields(item, is_cn=False),
            "description_cn": get_desc_fields(item, is_cn=True),
            "aspects": item.get("aspects", {}),
            "type": inferred_type,
            "icon": item.get("icon", item_id),
        }

        # Only books/tomes/journals etc. likely have xexts/xtriggers; but we can parse generically
        xexts = item.get("xexts")
        xexts_cn = item.get("xexts_cn")

        readings = parse_xexts_readings_only(xexts)
        readings_cn = parse_xexts_readings_only(xexts_cn)

        if readings:
            item_data["readings"] = readings
        if readings_cn:
            item_data["readings_cn"] = readings_cn

        xtriggers = item.get("xtriggers")
        xtriggers_cn = item.get("xtriggers_cn")

        results = parse_xtriggers(xtriggers, lookup_map)
        results_cn = parse_xtriggers(xtriggers_cn, lookup_map)

        if results:
            item_data["results"] = results
        if results_cn:
            item_data["results_cn"] = results_cn

        if group_key == "visitors":
            visitors_base.append(item_data)
        elif group_key == "incidents":
            if item['id'].split('.')[0] == 'incident':
                incidents_base.append(item_data)
        elif group_key == "weather":
            weather_entries.append(item)  # will convert below
        else:
            grouped[group_key].append(item_data)

    # 4. Merge visitors_embarking.json into visitors group
    #    Alignment heuristic: match by label/name or id prefix.
    if visitors_embarking:
        print(f"[Visitors] Merging {len(visitors_embarking)} embarking entries into base visitors...")
        # Build index by label for base visitors
        visitor_by_label = {}
        visitor_by_id = {}
        for v in visitors_base:
            visitor_by_label[v["name"]] = v
            visitor_by_id[v["id"]] = v

        for emb in visitors_embarking:
            emb_id = emb.get("id")
            emb_label = emb.get("Label", emb.get("label", ""))

            base = visitor_by_label.get(emb_label)
            if not base and emb_id:
                base = visitor_by_id.get(emb_id)

            if not base:
                # create a new visitor entry if no base found
                new_v = {
                    "id": emb_id,
                    "name": emb_label or "Unknown",
                    "name_cn": emb.get("Label_cn", emb.get("label_cn", emb_label or "Unknown")),
                    "description": get_desc_fields(emb, is_cn=False),
                    "description_cn": get_desc_fields(emb, is_cn=True),
                    "aspects": emb.get("aspects", {}),
                    "type": "visitor",
                    "icon": emb.get("icon", emb_id),
                }
                base = new_v
                visitors_base.append(base)

            xexts_emb = emb.get("xexts") or {}
            xexts_emb_cn = emb.get("xexts_cn") or {}

            if xexts_emb:
                base.setdefault("embarking_xexts", {}).update(xexts_emb)
                base['embarking_xexts']['dissatisfying_cn'] = xexts_emb_cn.get('dissatisfying', "")

    grouped["visitors"].extend(visitors_base)

    # 5. Convert weather_entries to grouped["weather"]
    weather_converted = []
    for w in weather_entries:
        wid = w.get("ID") or w.get("id")
        if not wid:
            continue
        w_data = {
            "id": wid,
            "name": w.get("Label", w.get("label", "Unknown")),
            "name_cn": w.get("Label_cn", w.get("label_cn", "Unknown")),
            "description": get_desc_fields(w, is_cn=False),
            "description_cn": get_desc_fields(w, is_cn=True),
            "aspects": w.get("aspects", {}),
            "type": "weather",
            "icon": w.get("icon", wid),
        }
        weather_converted.append(w_data)
    grouped["weather"].extend(weather_converted)

    # 6. Merge _visitactedaspects.json into incidents group
    if visitacted_entries:
        print(f"[Incidents] Merging {len(visitacted_entries)} visit-acted aspects into incidents...")

        # Build index: incident_suffix -> incident_obj
        # incident id usually looks like: "incident.opera.apollo"
        incident_suffix_map = {}
        for inc in incidents_base:
            inc_id = inc["id"]  # e.g. "incident.opera.apollo"
            if "." in inc_id:
                # strip leading "incident." -> "opera.apollo"
                suffix = inc_id.split(".", 1)[1]
            else:
                suffix = inc_id
            incident_suffix_map[suffix] = inc

        for va in visitacted_entries:
            va_id = va.get("id", "")
            va_label = va.get("Label", va.get("label", ""))  # e.g., "已阅读：摩根"
            va_desc = get_desc_fields(va, is_cn=False)
            va_desc_cn = get_desc_fields(va, is_cn=True)

            # expected pattern: acted.<visitor>.<...middle...>.<aspect>
            parts = va_id.split(".")
            if len(parts) < 4 or parts[0] != "acted":
                # unexpected pattern; skip or attach to all if you want fallback
                continue

            visitor_id = parts[1]                 # "morgen"
            aspect_id = parts[-1]                 # "grail"
            mid_parts = parts[2:-1]               # ["opera", "apollo"]
            incident_suffix = ".".join(mid_parts) # "opera.apollo"

            incident = incident_suffix_map.get(incident_suffix)
            if not incident:
                # no matching incident; skip
                continue

            # Attach reaction under incident["visitacted_reactions"][visitor_id][aspect_id]
            incident.setdefault("visitacted_reactions", {})
            visitor_map = incident["visitacted_reactions"].setdefault(visitor_id, {})
            visitor_map[aspect_id] = {
                "id": va_id,
                "label": va_label,
                "description": va_desc,
                "description_cn": va_desc_cn,
            }

    grouped["incidents"].extend(incidents_base)

    print(
        "[Group] tomes={tomes}, journals={journals}, "
        "aspecteditems={aspected}, skills={skills}, skills_r={skills_r}, "
        "incidents={incidents}, abilities={abilities}, visitors={visitors}, "
        "weather={weather}, others={others}".format(
            tomes=len(grouped["tomes"]),
            journals=len(grouped["journals"]),
            aspected=len(grouped["aspecteditems"]),
            skills=len(grouped["skills"]),
            skills_r=len(grouped["skills_r"]),
            incidents=len(grouped["incidents"]),
            abilities=len(grouped["abilities"]),
            visitors=len(grouped["visitors"]),
            weather=len(grouped["weather"]),
            others=len(grouped["others"]),
        )
    )

    # 7. Save grouped JSON
    output_file = os.path.join(OUTPUT_DIR, "boh_raw_data.json")
    data_to_save = {
        "game": "Book of Hours",
        "tomes": grouped["tomes"],
        "journals": grouped["journals"],
        "aspecteditems": grouped["aspecteditems"],
        "skills": grouped["skills"],
        "skills_r": grouped["skills_r"],
        "incidents": grouped["incidents"],
        "abilities": grouped["abilities"],
        "visitors": grouped["visitors"],
        "weather": grouped["weather"],
        "others": grouped["others"],
    }
    save_json(data_to_save, output_file)
    print(f"Saved grouped BOH data to {output_file}")
    print("Done.")


if __name__ == "__main__":
    run()