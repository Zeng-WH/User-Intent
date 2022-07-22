import json
import random
import copy

def convert_to_example(data, dial_ids):
    example_list = []
    for dial in data:
        dial_id = dial['id']
        if dial_ids is not None and dial_id not in dial_ids:
            continue
        for turn in dial['content']:
            ui = turn['用户意图']
            if ui == '':
                continue
            else:
                temp_json = {}
                temp_json['sentence'] = turn['用户']
                temp_json['label'] = ui
                example_list.append(temp_json)
    return example_list


def convert_to_example2(data, dial_ids):
    # 将用户的前两个utterance弄出来
    example_list= []
    for dial in data:
        dial_id = dial['id']
        if dial_ids is not None and dial_id not in dial_ids:
            continue
        utter_list = []
        for turn in dial['content']:
            prev_utter_list = copy.deepcopy(utter_list)
            ui = turn['用户意图']
            utter_list.append(turn['用户'])
            if ui == '':
                continue
            else:
                temp_json = {}
                temp_sent = prev_utter_list[-2:]
                temp_sent.append(turn['用户'])
                temp_sent = '[SEP]'.join(temp_sent)
                if len(temp_sent) > 256:
                    print('OOL')
                temp_json['sentence'] = temp_sent
                temp_json['label'] = ui
                example_list.append(temp_json)
    return example_list

def convert_to_example3(data, dial_ids):
    # 将前两个utterance弄出来
    example_list= []
    for dial in data:
        dial_id = dial['id']
        if dial_ids is not None and dial_id not in dial_ids:
            continue
        utter_list = []
        for turn in dial['content']:
            prev_utter_list = copy.deepcopy(utter_list)
            ui = turn['用户意图']
            utter_list.append(turn['用户'])
            utter_list.append(turn['客服'])
            if ui == '':
                continue
            else:
                temp_json = {}
                temp_sent = prev_utter_list[-2:]
                temp_sent.append(turn['用户'])
                temp_sent = '[SEP]'.join(temp_sent)
                if len(temp_sent) > 512:
                    print('OOL')
                temp_json['sentence'] = temp_sent
                temp_json['label'] = ui
                example_list.append(temp_json)
    return example_list




all_data = json.load(open('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/intent_data.json', 'r', encoding='utf-8'))
dial_ids=[dial['id'] for dial in all_data]
random.shuffle(dial_ids)
piece=len(dial_ids)//10
train_ids, dev_ids, test_ids=dial_ids[:8*piece], dial_ids[8*piece:9*piece], dial_ids[9*piece:]

train_example = convert_to_example3(all_data, train_ids)
dev_example = convert_to_example3(all_data, dev_ids)
test_example = convert_to_example3(all_data, test_ids)



with open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/datav3/train.json", "w", encoding='utf-8') as w:
    for item in train_example:
        w.write(json.dumps(item, ensure_ascii=False))
        w.write("\n")

with open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/datav3/dev.json", "w", encoding='utf-8') as w:
    for item in dev_example:
        w.write(json.dumps(item, ensure_ascii=False))
        w.write("\n")

with open("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/datav3/test.json", "w", encoding='utf-8') as w:
    for item in test_example:
        w.write(json.dumps(item, ensure_ascii=False))
        w.write("\n")

print("bupt")
