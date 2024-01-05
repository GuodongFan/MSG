from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import json
import random
from torch.autograd import Variable

random_seed=1
random.seed(random_seed)

def get_indices():
    with open('./data/mashup_name.json', 'r') as file:
        dataset = json.load(file)
    with open('./data/mashup_used_api.json', 'r') as f:
        mashup_used_api_ = json.load(f)

    filter_idx = []
    for idx, apis in enumerate(mashup_used_api_):
        if len(apis) >= 2:
            filter_idx.append(idx)

    #data_idx = list(range(len(dataset)))
    random.shuffle(dataset)
    split_num = int(len(dataset) / 10)
    test_idx = dataset[:split_num*3]
    train_idx = dataset[split_num * 3:]

    #train_idx, test_idx = train_test_split(dataset, test_size=0.3, random_state=random_seed)


    train_apis = set()
    oov_api = set()
    '''
    with open('./data/mashup_used_api.json', 'r') as f, open('./data/mashup_name.json', 'r') as f2:
        mashups = json.load(f2)
        mashup_apis = json.load(f)
        for idx, mashup in enumerate(mashups):
            if mashup in set(train_idx):
                apis = mashup_apis[idx]
                for api in apis:
                    train_apis.add(api)

    with open('./data/mashup_used_api.json', 'r') as f, open('./data/mashup_name.json', 'r') as f2:
        mashups = json.load(f2)
        mashup_apis = json.load(f)
        for idx, mashup in enumerate(mashups):
            if mashup in set(test_idx):
                apis = mashup_apis[idx]
                for api in apis:
                    if api not in train_apis:
                        print(api)
                        oov_api.add(api+'_api')
    '''
    print('oov {}'.format(len(oov_api)))
    return train_idx, test_idx, oov_api

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, str):
        x = self.model.encode(str, show_progress_bar=False,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

def sumary(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
    return Total_params, Trainable_params, NonTrainable_params

def attention_map(d, x_list, y_list):
    variables = x_list
    labels = y_list

    #df = pd.DataFrame(d.squeeze(1)[0], columns=variables, index=labels)

    fig = plt.figure(figsize=(20, 4))

    ax = fig.add_subplot(111)

    cax = ax.matshow(d.squeeze(1)[0:len(labels),:].tolist(), interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    fontdict = {'rotation': 60}
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xticklabels([''] + list(variables), fontdict=fontdict)
    ax.set_yticklabels([''] + list(labels))

    plt.show()

def cos_sim(a,b):
    """
    计算a，b向量的余弦相似度
    @param a: 1*m的向量
    @param b: n*m的矩阵
    @return: 1*n的值，每个样本的bi与a的余弦相似度
    """
    cos_result = np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
    return  cos_result

def sequence_mask(sequence_length, max_len=None):
    """
    Caution: Input and Return are VARIABLE.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    mask = seq_range_expand < seq_length_expand

    return mask