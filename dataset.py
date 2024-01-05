import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import spacy
import torchtext
from torchtext.vocab import GloVe
print(torchtext.__version__)
if torchtext.__version__ == '0.6.0':
    from torchtext import data
else:
    from torchtext.legacy import data
from utils import SequenceEncoder
import json
from torch.autograd import Variable
import random
random.seed(1024)

curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = curPath

MAX_CATE_LEN = 20

MAX_TGT_LEN = 11

MAX_SRC_LEN = 120

# encoder = SequenceEncoder()
#spacy_en = spacy.load('en_core_web_sm')
SOS = '<s>'
EOS = '<e>'
UNK = '<unk>'
PAD = '<pad>'

IPAD = 0
IEOS = 1

class AttrDict(dict):
    """ Access dictionary keys like attribute
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class APIDataset(Dataset):
    def __init__(self, data_file, rfile, rfile2,max_vocab_size=3000, train_dataset=None, word_dim=300, oov=None):
        self.max_vocab_size = max_vocab_size
        self.data_file = data_file
        self.train_data = train_dataset
        self.src, self.tgt, self.mashup_names, self.categories, self.retrieves, self.retrieves2 = self.load_sents2(rfile, rfile2, oov)

        with open('./data/mapping.json', 'r') as f:
            self.mapping = json.load(f)

        if train_dataset is None:
            self.TEXT = data.Field(sequential=True, unk_token=UNK, pad_token=PAD)
            # 从预训练的 vectors 中，将当前 corpus 词汇表的词向量抽取出来，构成当前 corpus 的 Vocab（词汇表）。
            corpus = self.src #+ (self.categories)
            self.TEXT.build_vocab(corpus, vectors=GloVe(name='6B', dim=word_dim), max_size=self.max_vocab_size)
            self.src_vocab = self.TEXT.vocab

            Label = data.Field(sequential=True, init_token=SOS, eos_token=EOS, unk_token=UNK, pad_token=PAD)
            Label.build_vocab(self.tgt, max_size=2000)
            #Label.build_vocab(self.tgt)

            self.tgt_vocab = Label.vocab
            #self.tgt_vocab.vectors = vector

            global IPAD, IEOS
            IPAD = self.tgt_vocab.stoi.get(PAD)
            IEOS = self.tgt_vocab.stoi.get(EOS)
        else:
            self.src_vocab = train_dataset.src_vocab
            self.tgt_vocab = train_dataset.tgt_vocab

    def __len__(self):
        return len(self.src)

    def translate(self, list):
        all_services = []
        for l in list:
            services = [self.tgt_vocab.itos[id] for id in l]
            all_services.append(services)
        return all_services

    def __getitem__(self, index):
        src_sent = self.src[index]
        tgt_sent = self.tgt[index]
        category = self.categories[index]
        if self.train_data is not None:
            retrieve = self.train_data.src[self.retrieves[index]]
            retrieve_category = self.train_data.categories[self.retrieves[index]]
        else:
            retrieve = self.src[self.retrieves[index]]
            retrieve_category = self.categories[self.retrieves[index]]
        #retrieve = self.retrieves[index].split(' ')
        retrieve2 = self.retrieves2[index]

        src_seq = self.tokens2ids(src_sent, self.src_vocab.stoi, append_BOS=False, append_EOS=False)
        tgt_seq = self.tokens2ids(tgt_sent, self.tgt_vocab.stoi, append_BOS=True, append_EOS=False)
        category_seq = self.tokens2ids(category, self.src_vocab.stoi, append_BOS=False, append_EOS=False)
        retrieve_seq = self.tokens2ids(retrieve, self.src_vocab.stoi, append_BOS=False, append_EOS=False)
        retrieve_category_seq = self.tokens2ids(retrieve_category, self.src_vocab.stoi, append_BOS=False, append_EOS=False)


        if len(src_seq) > MAX_SRC_LEN:
            src_seq = src_seq[:MAX_SRC_LEN]
        if len(tgt_seq) > MAX_TGT_LEN:
            tgt_seq = tgt_seq[:MAX_TGT_LEN -1]
            #tgt_seq.append(self.tgt_vocab.stoi[EOS])
        if len(category_seq) > MAX_CATE_LEN:
            category_seq = category_seq[:MAX_CATE_LEN]
        if len(retrieve_seq) > MAX_SRC_LEN:
            retrieve_seq = retrieve_seq[:MAX_SRC_LEN]
        if len(retrieve_category_seq) > MAX_CATE_LEN:
            retrieve_category_seq = retrieve_category_seq[:MAX_CATE_LEN]

        #random.shuffle(tgt_seq)
        retrieve2_seq = float(retrieve2[0])

        return src_sent, tgt_sent, src_seq, tgt_seq, self.mashup_names[index], category_seq, retrieve, retrieve_seq, retrieve2, retrieve2_seq, retrieve_category_seq

    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        new_tokens = []
        if append_BOS: new_tokens.append(SOS)
        new_tokens.extend(tokens)
        if append_EOS: new_tokens.append(EOS)
        seq.extend([token2id.get(token, token2id.get(UNK)) for token in new_tokens])
        return seq

    def load_apis(self, fpath):
        api_ids = []
        api_names = []
        api_descriptions = []
        df = pd.read_csv(fpath)
        with tqdm(df.iterrows(), total=df.shape[0]) as t:
            for index, row in t:
                api_id = row[0]
                api_name = row[1]
                api_description = row[3]
                api_ids.append(api_id)
                api_names.append(api_name)
                api_descriptions.append(api_description)
        return api_ids, api_names, api_descriptions

    def load_sents2(self,  rfile, rfile2, oov):
        mashup_names = []
        src_sents = []
        tgt_sents = []
        categories = []
        retrived = []
        retrived_apis = []

        select_mashups = []

        with open(rootPath + '/data/mashup_description.json', 'r') as f:
            mashup_description_ = json.load(f)
        with open(rootPath + '/data/mashup_category.json', 'r') as f:
            mashup_categories_ = json.load(f)
        with open(rootPath + '/data/mashup_name.json', 'r') as f:
            mashup_names_ = json.load(f)
        with open(rootPath + '/data/mashup_used_api.json', 'r') as f:
            mashup_used_api_ = json.load(f)
        #with open(rootPath + '/data/api_name.json', 'r') as f:
        #    api_names_ = json.load(f)


        num_api_1 = 0
        num_api_2 = 0
        average_num_api = 0
        api_mashup_dic = dict()
        for idx, apis in enumerate(mashup_used_api_):
            if len(apis) == 1:
                num_api_1 = num_api_1 + 1
            elif len(apis) >= 3:
                num_api_2 = num_api_2 + 1

            for api in apis:
                api_mashup = api_mashup_dic.get(api, None)
                if api_mashup == None:
                    mashup_set = set()
                    mashup_set.add(mashup_names_[idx])
                    api_mashup_dic[api] = {'api':api, 'mashup': mashup_set}
                else:
                    api_mashup['mashup'].add(mashup_names_[idx])

            average_num_api = average_num_api + len(apis)

        num_apis_1mashup = 0
        num_apis_2mashup = 0
        average_apis_mashup = 0
        for api, mashup in api_mashup_dic.items():
            mashup_set = mashup['mashup']
            if len(mashup_set) == 1:
                num_apis_1mashup = num_apis_1mashup + 1
            elif len(mashup_set) >= 2:
                num_apis_2mashup = num_apis_2mashup + 1

            average_apis_mashup += len(mashup_set)

        print('number of mashups: {}', len(mashup_names_))
        print('The number of mashups invoking only one service: {}', num_api_1)
        print('The number of mashups invoking at least two service: {}', num_api_2)
        print('The number of services: {}', len(api_mashup_dic))
        print('The number of services used in at least one mashup: {}', num_apis_1mashup)
        print('The number of services used in at least two mashup: {}', num_apis_2mashup)
        print('The average number of service in one mashup: {}', average_num_api/len(mashup_names_))

        # 把api的数据也加上 不管用删了...
        if oov is not None:

            with open(rootPath + '/data/api_name.json', 'r') as f, open(rootPath + '/data/api_description.json', 'r') as f2, open(rootPath + '/data/api_category.json', 'r') as f3:
                apis = json.load(f)
                descs = json.load(f2)
                cats = json.load(f3)
                for idx, api in enumerate(apis):
                    if api+'_api' in oov:
                        mashup_used_api_.append([api])
                        mashup_names_.append(api+'_api')
                        mashup_description_.append(descs[idx])
                        mashup_categories_.append(cats[idx])


        with open(rootPath + rfile, 'r', encoding='utf-8') as f:
            nl_retrived_ = f.readlines()

        with open(rootPath + rfile2, 'r', encoding='utf-8') as f:
            apis_retrived_ = f.readlines()

        filter_idx = []
        if oov is not None:
            self.data_file.extend(oov)
        for idx, mashup in enumerate(mashup_names_):
            if mashup in self.data_file:
                mashup_names.append(mashup)
                filter_idx.append(idx)

        for idx, mashup in enumerate(mashup_description_):
            if idx in filter_idx:
                src_sents.append(mashup)

        for idx, mashup in enumerate(mashup_categories_):
            if idx in filter_idx:
                categories.append(mashup)

        for idx, mashup in enumerate(mashup_used_api_):
            if idx in filter_idx:
                one_mashup = []
                for api in mashup:
                    one_mashup.append(api)
                #if idx %2 == 0:
                #    random.shuffle(one_mashup)
                tgt_sents.append(one_mashup)

        for idx, mashup in enumerate(nl_retrived_):
            retrived.append(int(mashup))

        for idx, apis in enumerate(apis_retrived_):
            all_apis = []
            for api in apis.split(','):
                if api != '' and api != '\n':
                    all_apis.append(api)
            retrived_apis.append(all_apis)

        '''     
        with open(rootPath + '/data/api_description.json', 'r') as f:
            if len(mashup_names) > 10000:
                for desc in f.readlines():
                    src_sents.append(desc)

        with open(rootPath + '/data/api_name.json', 'r') as f:
            if len(mashup_names) > 10000:
                for api in f.readlines():
                    tgt_sents.append(api)
        '''

        return src_sents, tgt_sents, mashup_names, categories, retrived, retrived_apis


def collate_fn(data):
    """
        Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
        We should build a custom collate_fn rather than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

        Args:
            data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
            - src_sents, tgt_sents: batch of original tokenized sentences
            - src_seqs, tgt_seqs: batch of original tokenized sentence ids
        Returns:
            - src_sents, tgt_sents (tuple): batch of original tokenized sentences
            - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
            - src_lens, tgt_lens (tensor): (batch_size)
    """

    def _pad_sequences(seqs, max_len):
        lens = [len(seq) for seq in seqs]
        #padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        padded_seqs = torch.full((len(seqs), max_len), getPAD()).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            if end > max_len:
                end = max_len
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_sents, tgt_sents, src_seqs, tgt_seqs, mashup_names, categories, retrieves, retrieve_seqs, retrieves2, retrieve2_seqs, retrieve_category_seq = zip(*data)

    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seqs, MAX_SRC_LEN)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs, MAX_TGT_LEN)
    category_seqs, category_lens = _pad_sequences(categories, MAX_CATE_LEN)
    retrieve_seqs, retrieve_lens = _pad_sequences(retrieve_seqs, MAX_SRC_LEN)
    retrieve_category_seqs, retrieve_category_lens = _pad_sequences(retrieve_category_seq, MAX_CATE_LEN)


    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0, 1)
    tgt_seqs = tgt_seqs.transpose(0, 1)
    category_seqs = category_seqs.transpose(0, 1)
    retrieve_seqs = retrieve_seqs.transpose(0, 1)
    retrieve_category_seqs = retrieve_category_seqs.transpose(0, 1)

    return src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, mashup_names, category_seqs, categories, category_lens, retrieves, retrieve_seqs, retrieve_lens, retrieves2, retrieve2_seqs, retrieve_lens, retrieve_category_seqs, retrieve_category_lens

def getPAD():
    return IPAD

def getEOS():
    return IEOS