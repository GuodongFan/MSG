import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from model.MSG.metric_new import metric, metric_string
from model.MSG.utils import get_indices

with open('./data/mashup_name.json', 'r') as file:
    mashups = json.load(file)

X_train, X_test = get_indices(mashups)

with open('./data/mashup_description.json', 'r') as f:
    mashup_description_ = json.load(f)

with open('./data/mashup_used_api.json', 'r') as f:
    mashup_apis_ = json.load(f)

filter_idx = []
train_code_list = []
train_api_list = []
test_code_list = []
test_api_list = []
for idx, mashup in enumerate(X_train):
    if mashup in mashups:
        filter_idx.append(idx)

for idx, desc in enumerate(mashup_description_):
    if idx in filter_idx:
        train_code_list.append((' ').join(desc).strip().rstrip())
        train_api_list.append(mashup_apis_[idx])
    else:
        test_code_list.append((' ').join(desc).strip().rstrip())
        test_api_list.append(mashup_apis_[idx])

corpus = train_code_list
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

def get_top_index(code, n):
    scores = bm25.get_scores(code)
    top_n = np.argsort(scores)[::-1][:n]
    return top_n

### 求metric
index = 0
top_k_list = [1, 5, 10]
ndcg_a = np.zeros(len(top_k_list))
recall_a = np.zeros(len(top_k_list))
ap_a = np.zeros(len(top_k_list))
pre_a = np.zeros(len(top_k_list))

result_list = []
result_api_list = []
for i in tqdm(range(len(test_code_list))):
    sim_idx = get_top_index(test_code_list[i].split(" "), n=1)[0]
    result_list.append(train_code_list[sim_idx])
    result_api_list.append(train_api_list[sim_idx])

    ### 求metric
    ndcg_, recall_, ap_, pre_ = metric_string(np.array(test_api_list[i]), np.array(train_api_list[sim_idx]),
                                              top_k_list)
    ndcg_a += ndcg_
    recall_a += recall_
    ap_a += ap_
    pre_a += pre_

    info = 'ApiLoss:' \
           'NDCG_A:%s\n' \
           'AP_A:%s\n' \
           'Pre_A:%s\n' \
           'Recall_A:%s\n' \
           % (
               (ndcg_a / index).round(6), (ap_a / index).round(6), (pre_a / index).round(6),
               (recall_a / index).round(6))

    print(info)
    # hr += getHR(1, apis, mval[1])
    # ndcg += getNDCG(1, np.array(apis), np.array(mval[1]))
    index += 1
    # loop.set_postfix(hr=hr/index, ndcg=ndcg/index)'''

df = pd.DataFrame(result_list)
df.to_csv("./data/BM25_test.csv", index=False, header=None)
df = pd.DataFrame(result_api_list)
df.to_csv("./data/BM25_test_api.csv", index=False, header=None)

result_list = []
result_api_list = []
for i in tqdm(range(len(train_code_list))):
    index = get_top_index(train_code_list[i].split(" "), n=5)
    for idx in index:
        if idx == i:
            continue
        result_list.append(train_code_list[idx])
        result_api_list.append(train_api_list[sim_idx])
        break

df = pd.DataFrame(result_list)
df.to_csv("./data/BM25_train.csv", index=False, header=None)
df = pd.DataFrame(result_api_list)
df.to_csv("./data/BM25_train_api.csv", index=False, header=None)

