import pickle
import faiss
import torch
import numpy as np
from nlgeval import compute_metrics
from tqdm import tqdm
from model.MSG.metric_new import metric, metric_string
from model.MSG.utils import sumary, attention_map, get_indices

from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd

from bert_whitening import sents_to_vecs, transform_and_normalize, normalize
from sklearn.model_selection import train_test_split

dim = 256

import json

with open('./data/mashup_name.json', 'r') as file:
    mashups_ = json.load(file)

X_train, X_test, oov = get_indices() #train_test_split(mashups, test_size=0.3, random_state=1024)

with open('./data/mashup_description.json', 'r') as f:
    mashup_description_ = json.load(f)

with open('./data/mashup_used_api.json', 'r') as f:
    mashup_apis_ = json.load(f)

with open('./data/mashup_category.json', 'r') as f:
    mashup_category_ = json.load(f)

# 把api的数据也加上 不管用删了...
# 把api的数据也加上 不管用删了...
if oov is not None:
    oov_apis = []
    with open('./data/api_name.json', 'r') as f, open('./data/api_description.json',
                                                                'r') as f2, open('./data/api_category.json',
                                                                                 'r') as f3:
        apis = json.load(f)
        descs = json.load(f2)
        cats = json.load(f3)
        for idx, api in enumerate(apis):
            if api+'_api' in oov:
                mashups_.append(api + "_api")
                oov_apis.append(api)
                mashup_apis_.append([api])
                mashup_description_.append(descs[idx])
                mashup_category_.append(cats[idx])

filter_idx = []
test_mashup_list = []
train_code_list = []
train_api_list = []
test_code_list = []
test_api_list = []
test_cate_list = []
train_cate_list = []


X_train.extend(oov)
for idx, mashup in enumerate(mashups_):
    if mashup in X_train:
        filter_idx.append(idx)
    else:
        test_mashup_list.append(mashup)

for idx, desc in enumerate(mashup_description_):
    if idx in filter_idx:
        train_code_list.append((' ').join(desc).strip().rstrip())
        train_api_list.append(mashup_apis_[idx])
        train_cate_list.append(mashup_category_[idx])
    else:
        test_code_list.append((' ').join(desc).strip().rstrip())
        test_api_list.append(mashup_apis_[idx])
        test_cate_list.append(mashup_category_[idx])


#with open('./data/api_description.json', 'r') as f:
#    apis = json.load(f)
#    for desc in apis:
#        train_code_list.append((' ').join(desc).strip().rstrip())

#df = pd.read_csv("./data/train.csv")
#train_code_list = df['MASHUP_DESC'].tolist()


#df = pd.read_csv("./data/test.csv")
#test_code_list = df['MASHUP_DESC'].tolist()

# with whitening
USE_WHITENING = True

#tokenizer = RobertaTokenizer.from_pretrained("./retrieve/codebert-base")
#model = RobertaModel.from_pretrained("./retrieve/bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("D://models/mymodel")
model = BertModel.from_pretrained("D://models/mymodel")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

def sim_jaccard(s1, s2):
    """jaccard相似度"""
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)  # 交集
    ret2 = s1.union(s2)  # 并集
    sim = 1.0 * len(ret1) / len(ret2)
    return sim

class Retrieval(object):
    def __init__(self):
        f = open('./model/MSG/retrieve/model/code_vector_whitening.pkl', 'rb')
        self.bert_vec = pickle.load(f)
        f.close()
        f = open('./model/MSG/retrieve/model/kernel.pkl', 'rb')
        self.kernel = pickle.load(f)
        f.close()
        f = open('./model/MSG/retrieve/model/bias.pkl', 'rb')
        self.bias = pickle.load(f)
        f.close()

        self.id2text = None
        self.vecs = None
        self.ids = None
        self.index = None

    def encode_file(self):
        all_texts = []
        all_ids = []
        all_vecs = []
        idx_vec = 0
        for i in range(len(train_code_list)):

            all_texts.append(train_code_list[i])
            all_ids.append(i)
            all_vecs.append(self.bert_vec[idx_vec].reshape(1,-1))
            idx_vec = idx_vec + 1
        all_vecs = np.concatenate(all_vecs, 0)
        id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
        self.id2text = id2text
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def single_query(self, code, ast, topK):
        body = sents_to_vecs([code], tokenizer, model)
        if USE_WHITENING == False:
            body = normalize(body)
            body = body[:, :dim]
        else:
            body = transform_and_normalize(body, self.kernel, self.bias)
        vec = body[[0]].reshape(1, -1).astype('float32')
        sim_dis, sim_idx = self.index.search(vec, topK)
        sim_idx = sim_idx[0].tolist()
        return sim_idx, sim_dis

if __name__ == '__main__':
    ccgir = Retrieval()
    print("Sentences to vectors")
    ccgir.encode_file()
    print("加载索引")
    ccgir.build_index(n_list=1)
    ccgir.index.nprob = 1
    sim_nl_list, sim_api_list = [], []
    data_list = []
    for i in tqdm(range(len(train_code_list))):

        sim_code, sim_dis = ccgir.single_query(train_code_list[i], train_code_list[i], topK=5)
        sim_nl_list.append(sim_code[1]) # 之前为啥改成2呢？
        sim_api_list.append(sim_dis[0][1])

    df = pd.DataFrame(sim_nl_list)
    df.to_csv("./data/Semantic_train.csv", index=False,header=None)
    df = pd.DataFrame(sim_api_list)
    df = (df - df.min()) / (df.max() - df.min())
    df.to_csv("./data/Semantic_train_api.csv", index=False, header=None)

    ### 求metric
    index = 0
    top_k_list = [1, 5, 10]
    ndcg_a = np.zeros(len(top_k_list))
    recall_a = np.zeros(len(top_k_list))
    ap_a = np.zeros(len(top_k_list))
    pre_a = np.zeros(len(top_k_list))

    sim_nl_list, sim_api_list = [], []
    data_list = []
    for i in tqdm(range(len(test_code_list))):

        sim_code,sim_dis = ccgir.single_query(test_code_list[i], test_code_list[i], topK=5)
        sim_nl_list.append(sim_code[0])
        sim_api_list.append(sim_dis[0][0])

        #if len(test_api_list[i]) <= 2:
        #    continue

        all_mashup_services = train_api_list[sim_code[0]]

        if test_mashup_list[i] in ['soundpushr', 'explore-travellr', 'gregs-alerts']:
            print(test_mashup_list[i])
            print(test_api_list[i])
            print(test_code_list[i])
            print(test_cate_list[i])
            print(all_mashup_services)
            print(train_code_list[sim_code[0]])
            print(train_cate_list[sim_code[0]])
        ### 求metric
        ndcg_, recall_, ap_, pre_ = metric_string(np.array(test_api_list[i]), np.array(train_api_list[sim_code[0]]), top_k_list)
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

    df = pd.DataFrame(sim_nl_list)
    df.to_csv("./data/Semantic_test.csv", index=False,header=None)

    df = pd.DataFrame(sim_api_list)

    df = (df - df.mean()) / df.std()
    df = (df - df.min()) / (df.max() - df.min())
    df.to_csv("./data/Semantic_test_api.csv", index=False,header=None)