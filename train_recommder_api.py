import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import APIDataset, collate_fn
from decoder import Decoder
from encoder import Encoder
from metric_new import metric
from utils import SequenceEncoder, sequence_mask
from utils import sumary, attention_map, get_indices, random_seed

topK = 10
top_k_list = [1, 5, 10]
epochs = 30
batch_size = 200

# 实验参数
num_layer = 1
hidden_dim = 200
dropout = 0.2
word_dim = 300
use_reinforcement_loss = False
use_retrieval = True

if use_reinforcement_loss:
    print('use reinforcement')

Train_Loss_list = []
Valid_Loss_list = []
Hit_ratio_list = []
NDCG_list = []

X_train, X_test, oov_api = get_indices()
train_dataset = APIDataset(X_train, '/data_api/Semantic_train.csv', '/data_api/Semantic_train_api.csv', oov=oov_api, word_dim=word_dim)
api_size = len(train_dataset.tgt_vocab)
test_dataset = APIDataset(X_test, '/data_api/Semantic_test.csv', '/data_api/Semantic_test_api.csv', train_dataset=train_dataset)

train_iter = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=collate_fn)

test_iter = DataLoader(dataset=test_dataset,
                       batch_size=batch_size,
                       shuffle=False,
                       num_workers=0,
                       collate_fn=collate_fn)

encoder_embedding = nn.Embedding(len(train_dataset.src_vocab), train_dataset.src_vocab.vectors.shape[1])
decoder_embedding = nn.Embedding(len(train_dataset.tgt_vocab), train_dataset.src_vocab.vectors.shape[1])
encoder_embedding.weight.data.copy_(train_dataset.src_vocab.vectors)
encoder_embedding.weight.requires_grad = True
decoder_embedding.weight.requires_grad = True

encoder = Encoder(embedding=encoder_embedding, hid_dim=hidden_dim, n_layers=num_layer, output_dim=hidden_dim,
                  dropout=dropout).cuda()
decoder = Decoder(embedding=decoder_embedding, hid_dim=hidden_dim * 2, n_layers=num_layer, output_dim=api_size,
                  dropout=dropout).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=1) #


encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-5)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=1e-5)
lr_scheduler_encoder = torch.optim.lr_scheduler.ExponentialLR(optimizer=encoder_optimizer,
                                                              gamma=0.99)  # gamma 越小 lr 下降越多
lr_scheduler_decoder = torch.optim.lr_scheduler.ExponentialLR(optimizer=decoder_optimizer, gamma=0.99)


def reward_function(decoded_seqs, ref_seqs, device):
    ref_seqs = ref_seqs.permute(1, 0)[:, 1:]
    scores = []
    for i in range(ref_seqs.shape[0]):
        try:
            ref_set = set(ref_seqs[i].tolist())
            ref_set.discard(1)
            ref_set.discard(0)
            ref_set.discard(2)
            ref_set.discard(3)

            dec_set = set(decoded_seqs[i].tolist())
            dec_set.discard(1)
            dec_set.discard(0)
            dec_set.discard(2)
            dec_set.discard(3)

            if (len(ref_set) == 0 ):
                print('ref is none')
                score = 0
            elif(len(dec_set) == 0):
                print('dec is none')
                score = 0
            else:
                score = (len(dec_set.intersection(ref_set))) / (
                    len(ref_set))  +(len(dec_set.intersection(ref_set)))/(len(dec_set))
        except Exception:
            score = 0
            print("Error occured at:")
            print("decoded_sents:", decoded_seqs[i])
            print("original_sents:", ref_seqs[i])
        scores.append(score)

    rouge_l_f1 = scores
    rouge_l_f1 = torch.tensor(rouge_l_f1, dtype=torch.float, device=device)
    return rouge_l_f1


def decode_one_batch_rl(greedy, hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens):
    # No teacher forcing for RL
    log_probs = []
    decode_ids = []
    dec_padding_mask = []
    mask_t = torch.ones(len(src_lens), dtype=torch.long).cuda()
    y_t = tgt[0]
    # there is at least one token in the decoded seqs, which is STOP_DECODING
    for di in range(10):
        y_t_1 = y_t
        # first we have coverage_t_1, then we have a_t
        prediction, hidden_n, attention_weights = decoder(y_t_1, hidden_n, outputs, outputs_category, src_lens,
                                                          category_lens)
        prediction = F.softmax(prediction, 1)
        if not greedy:
            # sampling
            multi_dist = Categorical(prediction)
            y_t = multi_dist.sample()
            log_prob = multi_dist.log_prob(y_t)
            log_probs.append(log_prob)

            y_t = y_t.squeeze(0).detach()
            dec_padding_mask.append(mask_t.detach().clone())
            mask_t[(mask_t == 1).long() + (y_t == 1).long() == 2] = 0

        else:
            # baseline
            # y_t = prediction.argmax(2).squeeze(0)
            y_t = prediction.topk(6).indices[:, :, 0]
            y_t = y_t.squeeze(0).detach()

        decode_ids.append(y_t)

    decode_ids = torch.stack(decode_ids, 1)

    if not greedy:
        dec_padding_mask = torch.stack(dec_padding_mask, 1).float()
        log_probs = torch.stack(log_probs, 1).squeeze(0).permute(1, 0).float() * dec_padding_mask
        dec_lens = dec_padding_mask.sum(1)

        log_probs = log_probs.sum(1) / dec_lens

    return decode_ids, log_probs


def infer_one_batch_ml(
        hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens):
    teacher_forcing_ratio = 0.5
    teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    predictions = torch.full([10, src_lens.shape[0], api_size], 0, dtype=torch.float32).cuda()

    step_losses = []
    for di in range(10):
        if di == 0:
            y_t_1 = tgt[di]
        else:
            y_t_1 = y_t
        # first we have coverage_t_1, then we have a_t
        prediction, hidden_n, attention_weights = decoder(y_t_1, hidden_n, outputs, outputs_category, src_lens,
                                                          category_lens)
        # if pointer_gen is True, the target will use the extend_vocab

        target = tgt[di + 1]
        # batch
        y_t = prediction.argmax(2).squeeze(0)
        predictions[di] = prediction

    loss = criterion(predictions.view(-1, predictions.shape[-1]), tgt[1:, :].contiguous().view(-1))
    # sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    # batch_avg_loss = sum_losses / 10

    # loss = loss + torch.mean(batch_avg_loss)

    return loss


def infer_one_batch_rl(hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens):
    # decode one batch
    decode_input = [hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens]
    sample_seqs, rl_log_probs = decode_one_batch_rl(False, *decode_input)
    with torch.autograd.no_grad():
        baseline_seqs, _ = decode_one_batch_rl(True, *decode_input)

    sample_reward = reward_function(sample_seqs, tgt, device='cuda:0')
    baseline_reward = reward_function(baseline_seqs, tgt, device='cuda:0')
    rl_loss = -(sample_reward - baseline_reward) * rl_log_probs
    rl_loss = torch.mean(rl_loss)
    batch_reward = torch.mean(sample_reward)

    return rl_loss, batch_reward


def decode_greedy(
        hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens, hidden_retrive, outputs_retrive,
        retrive_lens, outputs_retrive2, retrive2_lens, outputs_retrieve_category, retrieve_category_lens, tgt_apis, mashup_names):
    decode_seq = []
    for di in range(10):
        if di == 0:
            y_t_1 = tgt[di]
        else:
            y_t_1 = y_t

        # first we have coverage_t_1, then we have a_t
        prediction, hidden_n, attention_weights = decoder(y_t_1, hidden_n, outputs, outputs_category, src_lens,
                                                          category_lens)

        prediction2, hidden_n2, attention_weights2 = decoder(y_t_1, hidden_retrive, outputs_retrive,
                                                             outputs_retrieve_category,
                                                             retrive_lens, retrieve_category_lens)

        outputs_retrive2 = torch.tensor(outputs_retrive2)
        ones = torch.ones_like(outputs_retrive2)

        # 1651 写死了 (代表服务的个数)
        mul_ = (ones - outputs_retrive2).repeat(2004, 1).permute(1, 0).unsqueeze(0).cuda()
        if use_retrieval:
            y_t = (prediction+1*prediction2*mul_).argmax(2).squeeze(0)
        else:
            y_t = (prediction).argmax(2).squeeze(0)


        if di == 0:
            topk = torch.topk(prediction, 10, 2)

            '''all_mashup_services = train_dataset.translate(topk.indices.squeeze(0).cpu().numpy())
            for idx, apis in enumerate(tgt_apis):
                if len(apis) > 2:
                    tgt_set = set(tgt_apis[idx])
                    gen_set = set(all_mashup_services[idx])
                    results =  tgt_set.intersection(gen_set)
                    if len(results) > 2:
                        print(mashup_names[idx])
                        print(tgt_apis[idx])
                        print(all_mashup_services[idx])'''

            ndcg_, recall_, ap_, pre_ = metric(tgt.permute(1, 0)[:, 1:].cpu().numpy(),
                                               topk.indices.squeeze(0).cpu().numpy(), [1, 5, 10])

        decode_seq.append(y_t.tolist())

    return decode_seq, ndcg_, recall_, ap_, pre_


def test(epoch, num_epoch):
    encoder.eval()
    decoder.eval()
    epoch_loss = 0
    epoch_rl_loss = 0
    index = 0


    ndcg_g = np.zeros(len(top_k_list))
    recall_g = np.zeros(len(top_k_list))
    ap_g = np.zeros(len(top_k_list))
    pre_g = np.zeros(len(top_k_list))

    ndcg_r = np.zeros(len(top_k_list))
    recall_r = np.zeros(len(top_k_list))
    ap_r = np.zeros(len(top_k_list))
    pre_r = np.zeros(len(top_k_list))

    with tqdm(enumerate(test_iter), total=len(test_iter)) as loop:
        max_len = 10
        for batch_id, batch_data in loop:
            src = batch_data[2].cuda()
            tgt = batch_data[3].cuda()
            src_lens = batch_data[4]
            tgt_lens = batch_data[5]
            mashup_names = batch_data[6]
            category = batch_data[7].cuda()
            category_lens = batch_data[9]
            retrive = batch_data[11].cuda()
            retrive_lens = batch_data[12]
            retrive2 = batch_data[14]
            retrive2_lens = batch_data[15]
            retrieve_category_seqs = batch_data[16].cuda()
            retrieve_category_lens = batch_data[17]

            tgt_apis = batch_data[1]

            src_lens = Variable(torch.LongTensor(src_lens)).cuda()
            tgt_lens = Variable(torch.LongTensor(tgt_lens)).cuda()
            category_lens = Variable(torch.LongTensor(category_lens)).cuda()
            retrive_lens = Variable(torch.LongTensor(retrive_lens)).cuda()
            retrive2_lens = Variable(torch.LongTensor(retrive2_lens)).cuda()
            retrieve_category_lens = Variable(torch.LongTensor(retrieve_category_lens)).cuda()

            outputs_category, hidden_n_category = encoder(category, category_lens.data.tolist())
            outputs_retrieve_category, hidden_n_retrieve_category = encoder(retrieve_category_seqs,
                                                                            retrieve_category_lens.data.tolist())

            outputs_retrive, hidden_retrive = encoder(retrive, retrive_lens.data.tolist())
            outputs_retrive2 = retrive2
            outputs, hidden_n = encoder(src, src_lens.data.tolist())
            ml_loss = infer_one_batch_ml(hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens)
            rl_loss, reward = infer_one_batch_rl(hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens)
            if use_reinforcement_loss:
                loss = 0.8 * ml_loss + 0.2 * rl_loss
            else:
                loss = ml_loss
            epoch_loss += loss.item()
            epoch_rl_loss += rl_loss.item()

            index = index + 1

            decode_list, ndcg__, recall__, ap__, pre__ = decode_greedy(hidden_n, outputs, outputs_category, src_lens,
                                                                       category_lens, tgt, tgt_lens, hidden_retrive,
                                                                       outputs_retrive, retrive_lens, outputs_retrive2,
                                                                       retrive2_lens, outputs_retrieve_category,
                                                                       retrieve_category_lens, tgt_apis, mashup_names)
            all_mashup_services = train_dataset.translate(np.array(decode_list).transpose(1, 0).tolist())
            for idx, apis in enumerate(tgt_apis):
                if mashup_names[idx] in ['soundpushr', 'explore-travellr', 'gregs-alerts']:
                    tgt_set = set(tgt_apis[idx])
                    gen_set = set(all_mashup_services[idx])
                    results =  tgt_set.intersection(gen_set)

                    print(mashup_names[idx])
                    print(tgt_apis[idx])
                    print(all_mashup_services[idx])

                #if one_mashup == 'shahi':
                #
            ndcg_r += ndcg__
            recall_r += recall__
            ap_r += ap__
            pre_r += pre__

            ndcg_, recall_, ap_, pre_ = metric(tgt.permute(1, 0)[:, 1:].cpu().numpy(),
                                               np.array(decode_list).transpose(1, 0),
                                               top_k_list)
            ndcg_g += ndcg_
            recall_g += recall_
            ap_g += ap_
            pre_g += pre_

            # 更新信息
            loop.set_description(f'Ttest Epoch [{epoch}/{num_epoch}]')
            loop.set_postfix({'loss': epoch_loss / index, 'recall': recall_g / index})
        info = 'ApiLoss:' \
               'NDCG_G:%s\n' \
               'AP_G:%s\n' \
               'Pre_G:%s\n' \
               'Recall_G:%s\n' \
               % (
                   (ndcg_g / index).round(6), (ap_g / index).round(6),
                   (pre_g / index).round(6),
                   (recall_g / index).round(6))

        print(info)
        info = 'ApiLoss:' \
               'NDCG_R:%s\n' \
               'AP_R:%s\n' \
               'Pre_R:%s\n' \
               'Recall_R:%s\n' \
               % (
                   (ndcg_r / index).round(6), (ap_r / index).round(6),
                   (pre_r / index).round(6),
                   (recall_r / index).round(6))

        print(info)
    Valid_Loss_list.append(epoch_loss / index)


def train():
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        epoch_loss = 0
        epoch_rl_loss = 0
        epoch_ml_loss = 0
        index = 0
        with tqdm(enumerate(train_iter), total=len(train_iter)) as loop:
            max_len = 10
            for batch_id, batch_data in loop:
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                src = batch_data[2].cuda()
                tgt = batch_data[3].cuda()
                src_lens = batch_data[4]
                tgt_lens = batch_data[5]
                category = batch_data[7].cuda()
                category_lens = batch_data[9]
                retrive = batch_data[11].cuda()
                retrive_lens = batch_data[12]
                retrive2 = batch_data[14]
                retrive2_lens = batch_data[15]

                src_lens = Variable(torch.LongTensor(src_lens)).cuda()
                tgt_lens = Variable(torch.LongTensor(tgt_lens)).cuda()
                category_lens = Variable(torch.LongTensor(category_lens)).cuda()
                retrive_lens = Variable(torch.LongTensor(retrive_lens)).cuda()
                retrive2_lens = Variable(torch.LongTensor(retrive2_lens)).cuda()

                outputs_category, hidden_n_category = encoder(category, category_lens.data.tolist())
                outputs_retrive, hidden_n = encoder(retrive, retrive_lens.data.tolist())
                outputs, hidden_n = encoder(src, src_lens.data.tolist())
                ml_loss = infer_one_batch_ml(hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens)
                rl_loss, reward = infer_one_batch_rl(hidden_n, outputs, outputs_category, src_lens, category_lens, tgt, tgt_lens)

                if use_reinforcement_loss:
                    loss = 0.8 * ml_loss + 0.2 * rl_loss
                else:
                    loss = ml_loss

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                epoch_loss += loss.item()
                epoch_rl_loss += rl_loss.item()
                epoch_ml_loss += ml_loss.item()

                index = index + 1

                # 更新信息
                loop.set_description(
                    f'Train Epoch [{epoch}/{epochs}] rl {epoch_rl_loss / index} ml {epoch_ml_loss / index}')
                loop.set_postfix(loss=epoch_loss / index)

                # lr_scheduler_encoder.step()
                # lr_scheduler_decoder.step()
            print("第%d个epoch的学习率：%f" % (epoch, encoder_optimizer.param_groups[0]['lr']))

        Train_Loss_list.append(epoch_loss / index)
        test(epoch, epochs)


sumary(encoder)
sumary(decoder)
train()
