import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
import pandas as pd
import sys
from  model.MSG.utils import sumary, attention_map, get_indices
sys.path.append("./model/MSG")

print(sys.path)
from dataset import APIDataset, collate_fn

from sklearn.model_selection import train_test_split

MODEL_NAME = "D:/models/mymodel" # 本地模型文件

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

USE_WHITENING = True
N_COMPONENTS = 256
MAX_LENGTH = 100

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(name):
    #tokenizer = RobertaTokenizer.from_pretrained(name)
    tokenizer = BertTokenizer.from_pretrained(name)
    # model = RobertaModel.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    model = model.to(DEVICE)
    return tokenizer, model


def sents_to_vecs(sents, tokenizer, model):
    vecs = []
    with torch.no_grad():
        for sent in sents:
            if not isinstance(sent, str):
                continue
            # for sent in tqdm(sents):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LENGTH)
            inputs = inputs.to(DEVICE)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

            hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

            if POOLING == 'first_last_avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif POOLING == 'last_avg':
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif POOLING == 'last2avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise Exception("unknown pooling {}".format(POOLING))
            # output_hidden_state [batch_size, hidden_size]
            vec = output_hidden_state.cpu().numpy()[0]
            vecs.append(vec)
    #assert len(sents) == len(vecs)
    #print(len(vecs))
    vecs = np.array(vecs)
    return vecs


def compute_kernel_bias(vecs, n_components):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s**0.5))
    W = np.linalg.inv(W.T)
    W = W[:, :n_components]
    return W, -mu


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def main():
    print(f"Configs: {MODEL_NAME}-{POOLING}-{USE_WHITENING}-{N_COMPONENTS}.")
    tokenizer, model = build_model(MODEL_NAME)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))

    import json
    with open('./data/mashup_name.json', 'r') as file:
        mashups_ = json.load(file)

    X_train, X_test, oov = get_indices() #

    with open('./data/mashup_description.json', 'r') as f:
        mashup_description_ = json.load(f)

    with open('./data/mashup_category.json', 'r') as f:
        mashup_category_ = json.load(f)

    # 把api的数据也加上 不管用删了...
    oov_apis = []
    if oov is not None:
        with open('./data/api_name.json', 'r') as f,open('./data/api_description.json', 'r') as f2,open('./data/api_category.json', 'r') as f3:
            apis = json.load(f)
            descs = json.load(f2)
            cats = json.load(f3)
            for idx, api in enumerate(apis):
                if api+"_api" in oov:
                    mashup_description_.append(descs[idx])
                    oov_apis.append(api+"_api")
                    mashups_.append(api+"_api")

    filter_idx = []
    desc_list = []
    X_train.extend(oov_apis)

    for idx, mashup in enumerate(mashups_):
        if mashup in X_train:
            filter_idx.append(idx)

    for idx, desc in enumerate(mashup_description_):
        if idx in filter_idx:
            desc_list.append((' ').join(desc).strip().rstrip())

    # 加上API的信息
    #with open('./data/api_description.json', 'r') as f:
    #    apis = json.load(f)
   #    for desc in apis:
    #        desc_list.append((' ').join(desc).strip().rstrip())


    #df = pd.read_csv("./data/train.csv")
    #code_list = df['MASHUP_DESC'].tolist()
    print(len(desc_list))
    print("Transfer sentences to BERT vectors.")
    vecs_func_body = sents_to_vecs(desc_list, tokenizer, model) # [code_list_size, 768]
    if USE_WHITENING:
        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([
            vecs_func_body
        ], n_components=N_COMPONENTS)
        vecs_func_body = transform_and_normalize(vecs_func_body, kernel, bias) # [code_list_size, dim]
    else:
        print("Compute kernel and bias.")
        kernel, bias = compute_kernel_bias([
            vecs_func_body
        ], n_components=N_COMPONENTS)
        vecs_func_body = normalize(vecs_func_body)# [code_list_size, 768]
        vecs_func_body = vecs_func_body[:, :N_COMPONENTS]
    print(vecs_func_body.shape)
    import pickle
    f = open('./model/MSG/retrieve/model/code_vector_whitening.pkl', 'wb')
    pickle.dump(vecs_func_body, f)
    f.close()
    f = open('./model/MSG/retrieve/model/kernel.pkl', 'wb')
    pickle.dump(kernel, f)
    f.close()
    f = open('./model/MSG/retrieve/model/bias.pkl', 'wb')
    pickle.dump(bias, f)
    f.close()

if __name__ == "__main__":
    main()