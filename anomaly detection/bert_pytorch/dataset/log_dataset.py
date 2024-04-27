# import pickle
# from torch.utils.data import Dataset
# import torch
# import os
# import time
# import random
# import numpy as np
# from collections import defaultdict

# class LogDataset(Dataset):
#     def __init__(self, log_corpus, time_corpus, vocab, seq_len, 
#                                    parameter_embedding, template_and_parameter_embedding, 
#                                    corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.15):
#         """

#         :param corpus: log sessions/line
#         :param vocab: log events collection including pad, ukn ...
#         :param seq_len: max sequence length
#         :param corpus_lines: number of log sessions
#         :param encoding:
#         :param on_memory:
#         :param predict_mode: if predict
#         """
#         self.vocab = vocab
#         self.seq_len = seq_len

#         self.on_memory = on_memory
#         self.encoding = encoding

#         self.predict_mode = predict_mode
#         self.log_corpus = log_corpus
#         self.time_corpus = time_corpus
#         self.corpus_lines = len(log_corpus)

#         self.mask_ratio = mask_ratio
#         self.p_dict_embedding = parameter_embedding
#         self.t_and_p__dict_embedding = template_and_parameter_embedding



#     def __len__(self):
#         return self.corpus_lines

#     def __getitem__(self, idx):
#         k, t = self.log_corpus[idx], self.time_corpus[idx]
    
#         k_masked, k_label, t_masked, t_label, param_embedding = self.random_item(k, t)

#         # print(len(k))
#         # print(len(k_masked))
#         # print(k)
#         # print(k_masked)
#         # [CLS] tag = SOS tag, [SEP] tag = EOS tag
#         # vocab.pad_index = 0
#         # vocab.unk_index = 1
#         # vocab.sos_index = 3
#         # vocab.mask_index = 4
#         k = [self.vocab.sos_index] + k_masked
#         k_label = [self.vocab.pad_index] + k_label
#         # k_label = [self.vocab.sos_index] + k_label

#         t = [0] + t_masked
#         t_label = [self.vocab.pad_index] + t_label
#         return k, k_label, t, t_label, param_embedding

#     def random_item(self, k, t):
#         tokens = list(k)
#         output_label = []

#         time_intervals = list(t)
#         time_label = []

#         param_embedding = np.zeros([1,256],dtype=float)
#         for i, token in enumerate(tokens):
#             time_int = time_intervals[i]
#             prob = random.random()
#             # replace 15% of tokens in a sequence to a masked token
#             if prob < self.mask_ratio:
#                 # raise AttributeError("no mask in visualization")
#                 if self.predict_mode:
#                     t = np.zeros([1,256],dtype=float)
#                     param_embedding = np.concatenate((param_embedding, t), axis=0)
#                     tokens[i] = self.vocab.mask_index
#                     output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
#                     time_label.append(time_int)
#                     time_intervals[i] = 0
#                     continue

#                 prob /= self.mask_ratio

#                 # 80% randomly change token to mask token
#                 if prob < 0.8:
#                     tokens[i] = self.vocab.mask_index
#                     t = np.zeros([1,256],dtype=float)
#                     param_embedding = np.concatenate((param_embedding, t), axis=0)

#                 # 10% randomly change token to random token
#                 elif prob < 0.9:
#                     tokens[i] = random.randrange(len(self.vocab))
#                     if tokens[i] < 4:
#                         t = np.zeros([1,256],dtype=float)
#                         param_embedding = np.concatenate((param_embedding, t), axis=0)
#                     elif tokens[i] == 4:
#                         t = np.zeros([1,256],dtype=float)
#                         param_embedding = np.concatenate((param_embedding, t), axis=0)
#                     else:
#                         tokens[i] = random.randrange(len(self.t_and_p__dict_embedding))
#                         t = self.t_and_p__dict_embedding.get(int(tokens[i]), np.zeros([1,256],dtype=float))
#                         tokens[i] = self.vocab.stoi.get(tokens[i], self.vocab.unk_index)
#                         param_embedding = np.concatenate((param_embedding, t), axis=0)

#                 # 10% randomly change token to current token
#                 else:
#                     tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
#                     t = self.t_and_p__dict_embedding.get(int(token), np.zeros([1,256],dtype=float))
#                     param_embedding = np.concatenate((param_embedding, t), axis=0)

#                 output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

#                 time_intervals[i] = 0  # time mask value = 0
#                 time_label.append(time_int)

#             else:
#                 t = self.t_and_p__dict_embedding.get(int(token), np.zeros([1,256],dtype=float))
#                 param_embedding = np.concatenate((param_embedding, t), axis=0)
#                 tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
#                 output_label.append(0)
#                 time_label.append(0)

#         return tokens, output_label, time_intervals, time_label, param_embedding

#     # 规整函数，将不同长度的日志序列进行规整
#     def collate_fn(self, batch, percentile=100, dynamical_pad=True):
#         lens = [len(seq[0]) for seq in batch]

#         # find the max len in each batch
#         if dynamical_pad:
#             # dynamical padding
#             seq_len = int(np.percentile(lens, percentile))
#             if self.seq_len is not None:
#                 seq_len = min(seq_len, self.seq_len)
#         else:
#             # fixed length padding
#             seq_len = self.seq_len

#         output = defaultdict(list)
#         output["param_embedding"]=np.zeros([1, seq_len, 256], dtype=float)
#         for seq in batch:
#             bert_input = seq[0][:seq_len]
#             bert_label = seq[1][:seq_len]
#             time_input = seq[2][:seq_len]
#             time_label = seq[3][:seq_len]
#             param_embedding = seq[4]
#             if seq[4].shape[0] > seq_len:
#                 param_embedding = seq[4].array_split(seq_len, dim=0)[0]

#             padding = [self.vocab.pad_index for _ in range(seq_len - len(bert_input))]
#             bert_input.extend(padding), bert_label.extend(padding), time_input.extend(padding), time_label.extend(
#                 padding)
            
#             for _ in range(seq_len - param_embedding.shape[0]):
#                 param_embedding = np.concatenate((param_embedding, np.zeros([1,256],dtype=float)), axis=0)

#             time_input = np.array(time_input)[:, np.newaxis]
#             output["bert_input"].append(bert_input)
#             output["bert_label"].append(bert_label)
#             output["time_input"].append(time_input)
#             output["time_label"].append(time_label)
#             output["param_embedding"] = np.concatenate((np.expand_dims(param_embedding, axis=0), output["param_embedding"]), axis=0)

#         output["bert_input"] = torch.tensor(output["bert_input"], dtype=torch.long)
#         output["bert_label"] = torch.tensor(output["bert_label"], dtype=torch.long)
#         output["time_input"] = torch.tensor(output["time_input"], dtype=torch.float)
#         output["time_label"] = torch.tensor(output["time_label"], dtype=torch.float)
#         output["param_embedding"] = output["param_embedding"][1:]
#         output["param_embedding"] = torch.tensor(output["param_embedding"], dtype=torch.float32)

#         return output


import pickle
from torch.utils.data import Dataset
import torch
import os
import time
import random
import numpy as np
from collections import defaultdict

class LogDataset(Dataset):
    def __init__(self, log_corpus, time_corpus, vocab, seq_len, 
                                   parameter_embedding, template_and_parameter_embedding, 
                                   corpus_lines=None, encoding="utf-8", on_memory=True, predict_mode=False, mask_ratio=0.15):
        """

        :param corpus: log sessions/line
        :param vocab: log events collection including pad, ukn ...
        :param seq_len: max sequence length
        :param corpus_lines: number of log sessions
        :param encoding:
        :param on_memory:
        :param predict_mode: if predict
        """
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.encoding = encoding

        self.predict_mode = predict_mode
        self.log_corpus = log_corpus
        self.time_corpus = time_corpus
        self.corpus_lines = len(log_corpus)

        self.mask_ratio = mask_ratio
        self.p_dict_embedding = parameter_embedding
        self.t_and_p_dict_embedding = template_and_parameter_embedding



    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, idx):
        k, t = self.log_corpus[idx], self.time_corpus[idx]
    
        k_masked, k_label, t_masked, t_label, param_embedding = self.random_item(k, t)

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

        t = [0] + t_masked
        t_label = [self.vocab.pad_index] + t_label
        return k, k_label, t, t_label, param_embedding

    def random_item(self, k, t):
        tokens = list(k)
        output_label = []

        time_intervals = list(t)
        time_label = []

        param_embedding = np.zeros([1,256],dtype=float)
        # for i, token in enumerate(tokens):
        #     time_int = time_intervals[i]
        #     prob = random.random()
        #     # replace 15% of tokens in a sequence to a masked token
        #     if prob < self.mask_ratio:
        #         t = self.p_dict_embedding.get(int(token), np.zeros([1,256],dtype=float))
        #         param_embedding = np.concatenate((param_embedding, t), axis=0)
        #         # raise AttributeError("no mask in visualization")
        #         if self.predict_mode:
        #             tokens[i] = self.vocab.mask_index
        #             output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
        #             time_label.append(time_int)
        #             time_intervals[i] = 0
        #             continue

        #         prob /= self.mask_ratio

        #         # 80% randomly change token to mask token
        #         if prob < 0.8:
        #             tokens[i] = self.vocab.mask_index

        #         # 10% randomly change token to random token
        #         elif prob < 0.9:
        #             tokens[i] = random.randrange(len(self.vocab))

        #         # 10% randomly change token to current token
        #         else:
        #             tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        #         output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

        #         time_intervals[i] = 0  # time mask value = 0
        #         time_label.append(time_int)

        #     else:
        #         t = self.t_and_p_dict_embedding.get(int(token), np.zeros([1,256],dtype=float))
        #         param_embedding = np.concatenate((param_embedding, t), axis=0)
        #         tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
        #         output_label.append(0)
        #         time_label.append(0)

        for i, token in enumerate(tokens):
            time_int = time_intervals[i]
            prob = random.random()
            t = self.t_and_p_dict_embedding.get(int(token), np.zeros([1,256],dtype=float))
            param_embedding = np.concatenate((param_embedding, t), axis=0)
            # replace 15% of tokens in a sequence to a masked token
            if prob < self.mask_ratio:
                # raise AttributeError("no mask in visualization")
                if self.predict_mode:
                    tokens[i] = self.vocab.mask_index
                    output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))
                    time_label.append(time_int)
                    time_intervals[i] = 0
                    continue

                prob /= self.mask_ratio

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

                time_intervals[i] = 0  # time mask value = 0
                time_label.append(time_int)

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)
                time_label.append(0)

        return tokens, output_label, time_intervals, time_label, param_embedding

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
        output["param_embedding"]=np.zeros([1, seq_len, 256], dtype=float)
        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]
            time_input = seq[2][:seq_len]
            time_label = seq[3][:seq_len]
            param_embedding = seq[4]
            if seq[4].shape[0] > seq_len:
                param_embedding = np.split(seq[4], [seq_len], axis=0)[0]

            padding = [self.vocab.pad_index for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), time_input.extend(padding), time_label.extend(
                padding)
            
            for _ in range(seq_len - param_embedding.shape[0]):
                param_embedding = np.concatenate((param_embedding, np.zeros([1,256],dtype=float)), axis=0)

            time_input = np.array(time_input)[:, np.newaxis]
            output["bert_input"].append(bert_input)
            output["bert_label"].append(bert_label)
            output["time_input"].append(time_input)
            output["time_label"].append(time_label)
            output["param_embedding"] = np.concatenate((np.expand_dims(param_embedding, axis=0), output["param_embedding"]), axis=0)

        output["bert_input"] = torch.tensor(output["bert_input"], dtype=torch.long)
        output["bert_label"] = torch.tensor(output["bert_label"], dtype=torch.long)
        output["time_input"] = torch.tensor(output["time_input"], dtype=torch.float)
        output["time_label"] = torch.tensor(output["time_label"], dtype=torch.float)
        output["param_embedding"] = output["param_embedding"][1:]
        output["param_embedding"] = torch.tensor(output["param_embedding"], dtype=torch.float32)

        return output

