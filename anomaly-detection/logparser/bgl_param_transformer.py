import pickle
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
import os
import ast


class ParamTransformer:

    def __init__(self,
                 indir='./Drain_result/',
                 outdir='./Drain_result/',
                 label_dir='./Drain_result/',
                 hidden=256,
                 modelname='all-MiniLM-L6-v2',
                 logName=None):
        self.path = indir
        self.outpath = outdir
        self.hidden = hidden
        self.model = SentenceTransformer(modelname)
        # debug - start
        print(self.model.tokenizer.special_tokens_map)
        print(self.model.tokenizer.all_special_tokens)
        print(self.model.tokenizer.all_special_ids)
        print(
            self.model.tokenizer.tokenize(
                "This is a test sentence with three unknown tokens: [UNK] [UNK] [UNK]"
            ))  # Check how it's tokenized.
        # debug - end
        # debug
        print("Max Sequence Length:", self.model.max_seq_length)
        self.logName = logName
        self.labelpath = label_dir

    def model_embedding(self):
        import pickle
        with open(self.path + self.logName + "_templates.pkl", "rb") as f:
            template_df = pickle.load(f)

        template_data = defaultdict(list)
        template_2_id = dict()
        for eventid, template in zip(template_df["EventId"],
                                     template_df["EventTemplate"]):
            x = re.sub(r"<\*>", "[UNK]", template)
            template_data[eventid].append(x)
            assert len(template_data[eventid]) == 1
            template_2_id[template] = eventid
        template_unk = dict()
        for eventid, template in template_data.items():
            x = self.model.encode(
                sentences=template,
                output_value="token_embeddings",
            )
            template_unk[eventid] = x[0]
            template_unk[eventid] = (template_unk[eventid][self.model.tokenize(
                template)['input_ids'][0] == self.model.tokenizer.unk_token_id]
                                     ).cpu().numpy()
        # debug - start
        import pickle
        with open(os.path.join(self.path, "BGL.log_structured.pkl"),
                  'rb') as f:
            structured_df = pickle.load(f)
        structured_df = structured_df[["EventTemplate", "ParameterList"]]

        x = structured_df.apply(lambda row: template_unk[template_2_id[row[
            "EventTemplate"]]].shape[0],
                                axis=1) == structured_df.apply(
                                    lambda row: len(row["ParameterList"]),
                                    axis=1)
        assert (structured_df.apply(lambda row: template_unk[template_2_id[row[
            "EventTemplate"]]].shape[0],
                                    axis=1) == structured_df.apply(
                                        lambda row: len(row["ParameterList"]),
                                        axis=1)).all()
        print('Success!!!')
        param_list = []
        param_len = []
        for _ in structured_df["ParameterList"]:
            param_list.extend(_)
            param_len.append(len(_))
        param_vec = self.model.encode(
            sentences=param_list,
            output_value="sentence_embedding",
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        del param_list
        param_context_latent = []
        start_idx = 0
        for k, _ in enumerate(structured_df["EventTemplate"]):
            template_latent = template_unk[template_2_id[_]]
            param_latent = np.asarray(param_vec[start_idx:start_idx +
                                                param_len[k]])
            start_idx += param_len[k]

            assert len(template_latent) == len(param_latent)
            if len(param_latent) > 0:
                merge_latent = (template_latent + param_latent) / 2
                processed = torch.nn.functional.adaptive_avg_pool1d(
                    torch.from_numpy(merge_latent.mean(0, keepdims=True)),
                    256).squeeze(0).cpu().numpy()
            else:
                processed = None

            param_context_latent.append(processed)

        x_path = os.path.join(self.path, 'log_param_context.pkl')
        print(x_path)
        with open(x_path, 'wb') as f:
            pickle.dump(param_context_latent, f)
        print('Good...')
