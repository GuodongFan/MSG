import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import Attention
from enum import Enum
from utils import sequence_mask

class T(Enum):
    only_nmt = 1
    only_ret = 2
    only_cat = 4

    w_cat = 6
    w_cat_hierarchical = 7

class Decoder(nn.Module):
    def __init__(self, embedding=None, hid_dim=128, n_layers=1, output_dim=128, dropout=0.1):
        super().__init__()

        self.embedding = embedding
        self.emb_dim = embedding.embedding_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(self.emb_dim, hid_dim, n_layers, dropout=dropout)
        self.exe_t = T.w_cat


        #self.out = nn.Linear(hid_dim*2, output_dim)
        #hid_dim*2

        self.dropout = nn.Dropout(dropout)

        # attention
        self.attention = Attention(hid_dim*2)
        self.W_c = nn.Linear(self.hid_dim  + self.hid_dim,
                             self.hid_dim, bias=False)

        self.W_H = nn.Linear(hid_dim, 1, bias=True)

        self.W_S = nn.Linear(hid_dim*2, 1, bias=True)

        self.W_X = nn.Linear(self.emb_dim, 1, bias=True)

        self.W_s = nn.Linear(hid_dim, output_dim, bias=True)

        self.output_dim = output_dim

        if self.exe_t == T.only_nmt:
            self.out = nn.Linear(hid_dim*2, output_dim)
        elif self.exe_t == T.only_ret or self.exe_t == T.only_cat:
            self.out = nn.Linear(hid_dim, output_dim)
        elif self.exe_t == T.w_cat:
            self.out = nn.Linear(hid_dim*2 + hid_dim*1, output_dim)

        if self.n_layers > 1:
            self.layer_W = nn.Linear(self.n_layers, 1, bias=False)

    def predict(self, concat_input):
        if self.training:
            prediction = self.out(self.dropout(concat_input)).cuda().permute(1, 0, 2)
        else:
            prediction = self.out(concat_input).cuda().permute(1, 0, 2)

        return prediction

    def forward(self, input, hidden1, encoder_hiddens, outputs_category, src_lens, category_lens):
        batch_size = input.size(0)
        input = self.embedding(input.unsqueeze(0))
        attention_weights = None

        output, hidden1 = self.rnn(input, hidden1)
        concat_input = output
        #if self.is_attention:

        if self.n_layers > 1:
            hidden = self.layer_W(hidden1.permute(1, 2, 0)).permute(2, 0, 1)
        else:
            hidden = hidden1


        context_vector, attention_weights = self.attention(encoder_hiddens, hidden, src_lens)
        concat_input = torch.cat([context_vector.permute(1, 0, 2), hidden], -1)
        context_vector_c, attention_weights_c = self.attention(outputs_category, hidden, category_lens)


        if self.exe_t == T.only_nmt:
            prediction = self.predict(concat_input)
            return prediction.permute(1, 0, 2), hidden1, attention_weights


        elif self.exe_t == T.only_cat:
            concat_input = context_vector_c.permute(1, 0, 2) #, context_vector_r.permute(1, 0, 2) , context_vector_c.permute(1, 0, 2),

            prediction = self.predict(concat_input)

            return prediction.permute(1, 0, 2), hidden1, attention_weights

        elif self.exe_t == T.w_cat:
            concat_input = torch.cat([ concat_input, context_vector_c.permute(1, 0, 2) ], -1) #, context_vector_r.permute(1, 0, 2) , context_vector_c.permute(1, 0, 2),
            prediction = self.predict(concat_input)
            return prediction.permute(1, 0, 2), hidden1, attention_weights
        elif self.exe_t == T.w_cat_hierarchical:

            categoy_dist = self.W_s(context_vector_c)
            #retrive_dist = self.W_s(context_vector_r)

            #categoy_dist = torch.bmm(attention_weights_ca[:, :, 0].unsqueeze(2), categoy_dist)
            #api_dist = torch.bmm(attention_weights_ca[:, :, 1].unsqueeze(2), api_dist)

            #p_gen = torch.sigmoid(
            #    self.W_H(context_vector_cr) + self.W_S(concat_input).transpose(0, 1) + self.W_X(input.transpose(0, 1)))
            p_gen = torch.sigmoid(self.W_H(context_vector_c) + self.W_S(concat_input).transpose(0, 1) + self.W_X(input).transpose(0,1))
            batch_size = context_vector_c.size(0)
            p_gen_ = torch.ones(batch_size, self.n_layers, 1).cuda() - p_gen.cuda()
            #category_dist = torch.bmm(p_gen_.permute(0, 2, 1), self.W_s(context_vector_c))
            needs_dist = torch.bmm(p_gen.permute(0, 2, 1), self.out(concat_input.cuda()).permute(1, 0, 2))

            categoy_dist = torch.bmm(p_gen_.permute(0, 2, 1), categoy_dist)
            #api_dist = torch.bmm(p_gen_, api_dist)
            prediction =  categoy_dist + needs_dist
            return prediction.permute(1, 0, 2), hidden1, attention_weights

