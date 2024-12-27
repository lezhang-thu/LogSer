import pickle
import re
from sentence_transformers import SentenceTransformer
import argparse
import pandas as pd
import collections
import csv
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
from collections import defaultdict
from ekphrasis.classes.segmenter import Segmenter
import os


# 平均池化层
class PoolingLayer(nn.Module):
    def __init__(self, output_dim):
        super(PoolingLayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_dim)
    
    def forward(self, x):
        # 将输入 x 经过自适应平均池化层降维
        x = x.unsqueeze(1)  # 添加通道维度 (n, 1, x)
        x = self.pool(x)  # (n, 1, y)
        x = x.squeeze(1)  # 去掉中间一维 (n, y)
        return x


class ParamTransformer:
    def __init__(self, indir='./Drain_result/', outdir='./Drain_result/', label_dir='./Drain_result/', hidden=256, modelname='all-MiniLM-L6-v2', logName=None):
        self.path = indir
        self.outpath = outdir
        self.hidden = hidden
        self.model = SentenceTransformer(modelname, device='cuda:1')
        # self.model = SentenceTransformer(modelname)
        self.logName = logName
        self.labelpath = label_dir
        self.seg = Segmenter(corpus="english")

    def model_embedding(self):
        structured_df = pd.read_csv(self.path + self.logName + "_structured.csv")    
        structured_df["Label"] = structured_df["Label"].apply(lambda x: int(x != "-"))
        template_df = pd.read_csv(self.path + self.logName + "_templates.csv")

        template_data = defaultdict(list)
        for eventid, template in zip(template_df["EventId"], template_df["EventTemplate"]):
            template_data[eventid].append(self.seg.segment(template))

        parameter_normal = defaultdict(list)
        parameter_abnormal = defaultdict(list)
        for eventid, params, label in zip(structured_df["EventId"], structured_df["ParameterList"], structured_df["Label"]):
            if label == 0:
                parameter_normal[eventid].append(params)
            else:
                parameter_abnormal[eventid].append(params)

        #Sentences are encoded by calling model.encode()
        #Parameters
        # sentences – the sentences to embed
        # batch_size – the batch size used for the computation
        # show_progress_bar – Output a progress bar when encode sentences
        # output_value – Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        # convert_to_numpy – If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        # convert_to_tensor – If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        # device – Which torch.device to use for the computation
        # normalize_embeddings – If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        pool_layer = PoolingLayer(self.hidden)
        # 为每个模板的参数集合生成一个嵌入表达
        parameter_embeddings_normal = {}
        parameter_embeddings_abnormal = {}
        for eventid, params in parameter_normal.items():
            parameter_embeddings_normal[eventid] = self.model.encode(
                sentences=params, 
                convert_to_tensor =True, 
                show_progress_bar=True,
                output_value="sentence_embedding",
                normalize_embeddings=True 
            )
            x = pool_layer(parameter_embeddings_normal[eventid])
            parameter_embeddings_normal[eventid] = torch.mean(x, dim=0, keepdim=True)  
        for eventid, params in parameter_abnormal.items():
            parameter_embeddings_abnormal[eventid] = self.model.encode(
                sentences=params, 
                convert_to_tensor =True, 
                show_progress_bar=True,
                output_value="sentence_embedding",
                normalize_embeddings=True 
            )
            x = pool_layer(parameter_embeddings_abnormal[eventid])
            parameter_embeddings_abnormal[eventid] = torch.mean(x, dim=0, keepdim=True)  

        self.write_to_file(parameter_embeddings_normal, datatype="_embeddings_normal")
        self.write_to_file(parameter_embeddings_abnormal, datatype="_embeddings_abnormal")


        # 为每条模板生成一个嵌入表达，并与参数嵌入表达相加
        template_and_parameter_embeddings_normal = {}
        for eventid, template in template_data.items():
            if eventid in parameter_embeddings_normal:
                template_and_parameter_embeddings_normal[eventid] = self.model.encode(
                    sentences=template, 
                    convert_to_tensor =True, 
                    show_progress_bar=True,
                    output_value="sentence_embedding",
                    normalize_embeddings=True 
                )
                template_and_parameter_embeddings_normal[eventid] = pool_layer(template_and_parameter_embeddings_normal[eventid]) + parameter_embeddings_normal[eventid]
        template_and_parameter_embeddings_abnormal = {}
        for eventid, template in template_data.items():
            if eventid in parameter_embeddings_abnormal:
                template_and_parameter_embeddings_abnormal[eventid] = self.model.encode(
                    sentences=template, 
                    convert_to_tensor =True, 
                    show_progress_bar=True,
                    output_value="sentence_embedding",
                    normalize_embeddings=True 
                )
                template_and_parameter_embeddings_abnormal[eventid] = pool_layer(template_and_parameter_embeddings_abnormal[eventid]) + parameter_embeddings_abnormal[eventid]

        self.write_to_file(template_and_parameter_embeddings_normal, datatype="_with_template_embeddings_normal")
        self.write_to_file(template_and_parameter_embeddings_abnormal, datatype="_with_template_embeddings_abnormal")

    def write_to_file(self, data, datatype="tempdata"):
        with open(self.path+self.logName+datatype+'.pickle', 'wb') as f:
            pickle.dump(data, f)

    def read_from_file(self, datatype="tempdata"):
        with open(self.path+self.logName+datatype+'.pickle', 'rb') as f:
            data = pickle.load(f)
        return data

# import pickle
# import re
# from transformers import AutoTokenizer, AutoModel
# from bpe import Encoder
# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from d2l import torch as d2l
# from collections import defaultdict
# import torch.nn.functional as F
# import os
# import tqdm


# # 平均池化层
# class PoolingLayer(nn.Module):
#     def __init__(self, output_dim):
#         super(PoolingLayer, self).__init__()
#         self.pool = nn.AdaptiveAvgPool1d(output_dim)
    
#     def forward(self, x):
#         # 将输入 x 经过自适应平均池化层降维
#         x = x.unsqueeze(1)  # 添加通道维度 (n, 1, x)
#         x = self.pool(x)  # (n, 1, y)
#         x = x.squeeze(1)  # 去掉中间一维 (n, y)
#         return x
    
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# class ParamTransformer:
#     def __init__(self, indir='./Drain_result/', outdir='./Drain_result/', label_dir='./Drain_result/', hidden=256, 
#                  modelname='all-MiniLM-L6-v2', logName=None,vocab_size = 150, pct_bpe=0.65):
#         self.path = indir
#         self.outpath = outdir
#         self.hidden = hidden
#         self.labelpath = label_dir
#         self.param_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/'+modelname)
#         self.model = AutoModel.from_pretrained('sentence-transformers/'+modelname)
#         self.logName = logName
#         self.vocab_size = vocab_size
#         self.pct_bpe = pct_bpe
#         self.template_tokenizer = self.tokenizer_model()   
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     def param_embedding(self, sequence):
#         encoded_input = self.param_tokenizer(sequence, padding=True, truncation=True, return_tensors='pt').to(self.device)
#         with torch.no_grad():
#             model_output = self.model(**encoded_input)
#         sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#         sentence_embeddings  = F.normalize(sentence_embeddings, p=2, dim=1)
#         return sentence_embeddings

#     def template_embedding(self, sequence):
#         encoded_input=defaultdict(list)
#         input_id = next(self.template_tokenizer.transform(sequence))
#         encoded_input['input_ids'].append(input_id)
#         token_type_ids = [0]*len(input_id)
#         encoded_input['token_type_ids'].append(token_type_ids)
#         attention_mask = [1]*len(input_id)
#         encoded_input['attention_mask'].append(attention_mask)

#         encoded_input['input_ids'] = torch.tensor(encoded_input['input_ids'], dtype=torch.long).to(self.device)
#         encoded_input['token_type_ids'] = torch.tensor(encoded_input['token_type_ids'], dtype=torch.long).to(self.device)  
#         encoded_input['attention_mask'] = torch.tensor(encoded_input['attention_mask'], dtype=torch.long).to(self.device)
#         with torch.no_grad(): 
#             model_output = self.model(**encoded_input)
#             sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
#             sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
#         return sentence_embeddings

#     def tokenizer_model(self):
#         structured_df = pd.read_csv(self.path + self.logName + "_templates.csv") 
#         train_corpus = structured_df['EventTemplate'].tolist()
#         encoder = Encoder(self.vocab_size, pct_bpe=self.pct_bpe)  # params chosen for demonstration purposes
#         encoder.fit(train_corpus)
#         return encoder

#     def model_embedding(self):
#         self.model.to(self.device)
#         structured_df = pd.read_csv(self.path + self.logName + "_structured.csv")
#         template_df = pd.read_csv(self.path + self.logName + "_templates.csv")
#         label_df = pd.read_csv(self.labelpath+"anomaly_label.csv")
#         dict_label = defaultdict(bool)
#         for idx, row in label_df.iterrows():
#             dict_label[row['BlockId']] = row['Label'] == 'Normal'

#         template_data = defaultdict(list)
#         for eventid, template in zip(template_df["EventId"], template_df["EventTemplate"]):
#             template_data[eventid].append(template)

#         parameter_normal = defaultdict(list)
#         parameter_abnormal = defaultdict(list)
#         for eventid, params, content in zip(structured_df["EventId"], structured_df["ParameterList"], structured_df["Content"]):
#             blockid = re.search(r'(blk_-?\d+)', content)[0]
#             if dict_label[blockid]:
#                 parameter_normal[eventid].append(params)
#             else:
#                 parameter_abnormal[eventid].append(params)
 
#         pool_layer = PoolingLayer(self.hidden)
#         # 为每个模板的参数集合生成一个嵌入表达
#         parameter_embeddings_normal = {}
#         parameter_embeddings_abnormal = {}
    
#         print("normal parameter embedding…………")
#         for eventid, params in parameter_normal.items():
#             param_list = [params[i:i + 256] for i in range(0, len(params), 256)]
#             parameter_embeddings_normal[eventid] = torch.zeros([1, 256], dtype=torch.float).to(self.device)
#             for i in tqdm.tqdm(param_list, total=len(param_list)):
#                 sentence_embeddings = self.param_embedding(i)
#                 x = pool_layer(sentence_embeddings)
#                 parameter_embeddings_normal[eventid] = torch.cat((parameter_embeddings_normal[eventid], x), dim=0)
#             parameter_embeddings_normal[eventid] = torch.mean(x, dim=0, keepdim=True)
#         print("normal parameter embedding done") 

#         # 为每条模板生成一个嵌入表达，并与参数嵌入表达相加
#         print("normal template embedding…………")
#         template_and_parameter_embeddings_normal = {}
#         for eventid, template in tqdm.tqdm(template_data.items(), total=len(template_data)):
#             if eventid in parameter_embeddings_normal:
#                 template_and_parameter_embeddings_normal[eventid] = self.template_embedding(template)
#                 template_and_parameter_embeddings_normal[eventid] = pool_layer(template_and_parameter_embeddings_normal[eventid]) + parameter_embeddings_normal[eventid]
#         print("normal template embedding done") 

#         self.write_to_file(parameter_embeddings_normal, datatype="_embeddings_normal")
#         self.write_to_file(template_and_parameter_embeddings_normal, datatype="_with_template_embeddings_normal")

#         del parameter_embeddings_normal
#         del template_and_parameter_embeddings_normal

#         print("abnormal parameter embedding…………")
#         for eventid, params in parameter_abnormal.items():
#             param_list = [params[i:i + 256] for i in range(0, len(params), 256)]
#             parameter_embeddings_abnormal[eventid] = torch.zeros([1, 256], dtype=torch.float).to(self.device)
#             for i in tqdm.tqdm(param_list, total=len(param_list)):
#                 sentence_embeddings = self.param_embedding(i)
#                 x = pool_layer(sentence_embeddings)
#                 parameter_embeddings_abnormal[eventid] = torch.cat((parameter_embeddings_abnormal[eventid], x), dim=0)
#             parameter_embeddings_abnormal[eventid] = torch.mean(x, dim=0, keepdim=True)
#         print("abnormal parameter embedding done") 

#         print("abnormal template embedding…………")
#         template_and_parameter_embeddings_abnormal = {}
#         for eventid, template in tqdm.tqdm(template_data.items(), total=len(template_data)):
#             if eventid in parameter_embeddings_abnormal:
#                 template_and_parameter_embeddings_abnormal[eventid] = self.template_embedding(template)
#                 template_and_parameter_embeddings_abnormal[eventid] = pool_layer(template_and_parameter_embeddings_abnormal[eventid]) + parameter_embeddings_abnormal[eventid]
#         print("abnormal template embedding done") 

#         self.write_to_file(parameter_embeddings_abnormal, datatype="_embeddings_abnormal")
#         self.write_to_file(template_and_parameter_embeddings_abnormal, datatype="_with_template_embeddings_abnormal")
        
#     def write_to_file(self, data, datatype="tempdata"):
#         with open(self.path+self.logName+datatype+'.pickle', 'wb') as f:
#             pickle.dump(data, f)

#     def read_from_file(self, datatype="tempdata"):
#         with open(self.path+self.logName+datatype+'.pickle', 'rb') as f:
#             data = pickle.load(f)
#         return data
