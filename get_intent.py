import json
#from baseline.configv3 import global_config as cfg
import re

# 暂时忽略意图的噪声


processed_data = json.load(open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/Training_Data_and_Task_Description/Track2_data/processed_data.json", 'r', encoding='utf-8'))

user_intent = [
    "求助-查询",
    "求助-故障",
    "提供信息",
    "投诉反馈",
    "取消",
    "询问",
    "请求重复",
    "主动确认",
    "被动确认",
    "否认",
    "问候",
    "再见",
    "客套",
    "其他"
]

user_num = {}
for item in user_intent:
    user_num[item] = 0
noise = []

intent_set = set()
for item in processed_data:
    for turn in item['content']:
        temp_ui = []
        ui = turn['用户意图']
        ui_list = []
        if '(' in ui:
            info = re.findall(r'\((.*?)\)', ui)
            for i in range(10):
                if '(' not in ui:
                    break
                else:
                    for e in info:
                        ui = ui.replace('('+e+')','')
        ui = ui.split(',')
        for intent in ui:
            if '(' in intent:
                idx = intent.index('(')
                intent = intent[:idx]
            if intent in user_intent:
                user_num[intent] += 1
                temp_ui.append(intent)
            else:
                noise.append(intent)
            intent_set.add(intent)
        temp_ui = ','.join(temp_ui)
        turn['用户意图'] = temp_ui

with open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/intent_data.json", "w", encoding='utf-8') as w:
    json.dump(processed_data, w, indent=2, ensure_ascii=False)



print("bupt")