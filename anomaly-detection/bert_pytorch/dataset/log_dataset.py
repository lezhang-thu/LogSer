import pickle
from torch.utils.data import Dataset
import torch
import os
import time
import random
import numpy as np
from collections import defaultdict


class LogDataset(Dataset):

    def __init__(
        self,
        log_corpus,
        idx_seq_corpus,
        vocab,
        seq_len,
        encoding="utf-8",
        on_memory=True,
        predict_mode=False,
        mask_ratio=0.15,
        param_context=None,
    ):
        """

        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.encoding = encoding

        self.predict_mode = predict_mode
        self.size = len(log_corpus)
        self.log_corpus = log_corpus
        self.idx_seq_corpus = idx_seq_corpus

        self.mask_ratio = mask_ratio
        assert type(param_context) is list
        self.param_context = param_context
        self.np_pad = np.zeros((256,), dtype=np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        k_masked, k_label, param_embedding = self.random_item(
            self.log_corpus[idx], self.idx_seq_corpus[idx])
        # debug
        #print('#' * 20)
        #print('len(k_masked): {}'.format(len(k_masked)))
        #print('len(param_embedding): {}'.format(len(param_embedding)))
        #exit(0)

        # print(len(k))
        # print(len(k_masked))
        # print(k)
        # print(k_masked)
        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        # vocab.pad_index = 0
        # vocab.unk_index = 1
        # vocab.sos_index = 3
        # vocab.mask_index = 4
        k = [self.vocab.sos_index] + k_masked
        k_label = [self.vocab.pad_index] + k_label
        # k_label = [self.vocab.sos_index] + k_label

        return k, k_label, param_embedding

    def random_item(self, k, t):
        tokens = list(k)
        output_label = []
        idx_seq_intervals = list(t)

        param_embedding = [self.np_pad]
        for k_idx, token in enumerate(tokens):
            t = self.param_context[int(idx_seq_intervals[k_idx])]

            prob = random.random()
            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                t = self.np_pad
                if self.predict_mode:
                    tokens[k_idx] = self.vocab.mask_index
                    output_label.append(
                        self.vocab.stoi.get(token, self.vocab.unk_index))
                else:
                    prob /= self.mask_ratio
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[k_idx] = self.vocab.mask_index
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[k_idx] = random.randrange(len(self.vocab))
                    # 10% randomly change token to current token
                    else:
                        tokens[k_idx] = self.vocab.stoi.get(
                            token, self.vocab.unk_index)
                    output_label.append(
                        self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[k_idx] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(self.vocab.pad_index)
            param_embedding.append(self.np_pad if t is None else t)

        return tokens, output_label, param_embedding

    # 规整函数，将不同长度的日志序列进行规整
    def collate_fn(self, batch, percentile=100, dynamical_pad=True):
        lens = [len(seq[0]) for seq in batch]

        # find the max len in each batch
        if dynamical_pad:
            # dynamical padding
            seq_len = int(np.percentile(lens, percentile))
            if self.seq_len is not None:
                seq_len = min(seq_len, self.seq_len)
        else:
            # fixed length padding
            seq_len = self.seq_len

        output = defaultdict(list)
        output["param_embedding"] = []
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]
            param_embedding = seq[2][:seq_len]

            padding = [
                self.vocab.pad_index for _ in range(seq_len - len(bert_input))
            ]
            bert_input.extend(padding), bert_label.extend(padding)
            x = len(param_embedding)
            for _ in range(seq_len - x):
                param_embedding.append(self.np_pad)

            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["param_embedding"].append(np.stack(param_embedding, 0))

        output["bert_input"] = torch.tensor(output["bert_input"]).long()
        output["bert_label"] = torch.tensor(output["bert_label"]).long()
        output["param_embedding"] = torch.from_numpy(
            np.stack(output["param_embedding"], 0)).float()

        return output
