from utils import *

_, test, _ = get_indices()

with open('./data/mashup_name.json', 'r') as f:
    mashup_names = json.load(f)
with open('./data/mashup_used_api.json', 'r') as f:
    mashup_used_api = json.load(f)

api_num = []
idx = 0
for _, mashup in enumerate(mashup_names):
    if mashup in test and idx < 1000:
        api_num.append(str(len(mashup_used_api[idx]))+'\n')
        idx = idx + 1

with open('./test.out', 'w') as file:
    file.writelines(api_num)
