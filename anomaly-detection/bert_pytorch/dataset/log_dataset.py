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
        self.idx2log = param_context
        print('self.vocab.pad_index: {}'.format(self.vocab.pad_index))
        print('self.vocab.sos_index: {}'.format(self.vocab.sos_index))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        k_masked, k_label, context = self.random_item(self.log_corpus[idx],
                                                      self.idx_seq_corpus[idx])
        k = [self.vocab.sos_index] + k_masked
        k_label = [self.vocab.pad_index] + k_label
        context = [None] + context

        return k, k_label, context

    def random_item(self, k, t):
        tokens = list(k)
        output_label = []
        idx_seq = list(t)

        ## logs' embeddings (logs in idx_seq)
        #x = self.st.encode(
        #    sentences=[self.idx2log[int(_)] for _ in idx_seq],
        #    output_value="sentence_embedding",
        #    convert_to_numpy=True,
        #)
        #log_seq_embed = torch.nn.functional.adaptive_avg_pool1d(
        #    torch.from_numpy(np.asarray(x)), 256).numpy()

        for k_idx, token in enumerate(tokens):
            if random.random() < self.mask_ratio:
                tokens[k_idx] = self.vocab.mask_index
                output_label.append(
                    self.vocab.stoi.get(token, self.vocab.unk_index))
            else:
                tokens[k_idx] = self.vocab.stoi.get(token,
                                                    self.vocab.unk_index)
                output_label.append(self.vocab.pad_index)
        context = [self.idx2log[int(_)] for _ in idx_seq]
        return tokens, output_label, context

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
        output["context"] = []
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]
            context = seq[2][:seq_len]

            padding = [
                self.vocab.pad_index for _ in range(seq_len - len(bert_input))
            ]
            bert_input.extend(padding), bert_label.extend(padding)
            x = len(context)
            for _ in range(seq_len - x):
                context.append(None)

            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["context"].append(context)

        output["bert_input"] = torch.tensor(output["bert_input"]).long()
        output["bert_label"] = torch.tensor(output["bert_label"]).long()
        return output
