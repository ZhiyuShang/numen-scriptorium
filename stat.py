import os
import json

def roman_to_int(s):
    if not s: return -1
    s = s.upper()
    if s == '0': return 0
    rom_val = {'O': 0 ,'I': 1, 'V': 5, 'X': 10}
    int_val = 0
    try:
        for i in range(len(s)):
            if i > 0 and rom_val[s[i]] > rom_val[s[i - 1]]:
                int_val += rom_val[s[i]] - 2 * rom_val[s[i - 1]]
            else:
                int_val += rom_val[s[i]]
        return int_val
    except KeyError:
        return -1
    
def replacing(key_re,item):
    origins_dict = {
        '琥珀': 'Amber',
        '尚未诞生': 'Not yet born',
        "未知":'Unknown',
        '肉源':'Flesh',
        '虚源':'Nowhere',
        '血源':'Blood',
        '光源':'Light',
        '石源':'Stone'
    }
    try:
        ans = origins_dict.get(key_re,key_re)
    except:
        breakpoint()
    return ans

def merge_hours_stat():
    file_path = os.path.join('data','hours_merged.json')
    if not os.path.exists(file_path):
        print('file not exists')
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print('number of hours',len(data))
        ids = set()
        sourceIds = set()
        origins = set()
        factions = set()
        hour_numbers = set()
        targets= set()
        for item in data:
            if item['id'] in ids:
                print(f'Duplicate id {item["id"]}')
            if item['sourceId'] in sourceIds:
                print(f'Duplicate sourceId {item["sourceId"]}')
            ids.add(item['id'])
            sourceIds.add(item['sourceId'])

            sid = item.get('sourceId', '')
            if sid.startswith('hour_'):
                parts = sid.split('_')
                if len(parts) > 1:
                    roman = parts[1]
                    num = roman_to_int(roman)
                    if 0 <= num <= 40:
                        if num in hour_numbers:
                            print(f'Duplicate hour number: {num} (sourceId: {sid})')
                        hour_numbers.add(num)
                    else:
                        print(f'Invalid hour number: {num} (sourceId: {sid})')
            if type(item['origin']) != list:
                item['origin'] = [item['origin']]
            for ori in item['origin']:
                target = targets.add(replacing(ori,item))
                origins.add(ori)
            
            for fra in item['factions']:
                factions.add(fra)

        print('origins',origins)
        print('targets',targets)
        print('factions',factions)

        missing_nums = [i for i in range(41) if i not in hour_numbers]
        if missing_nums:
            print(f'Missing hour numbers (0-40): {missing_nums}')




if __name__ == '__main__':
    merge_hours_stat()