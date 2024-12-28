import os
import numpy as np
import pickle
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from bert_pytorch.dataset import WordVocab
from bert_pytorch.dataset import LogDataset
from bert_pytorch.dataset.sample import fixed_window
from sentence_transformers import SentenceTransformer

threshold = 0.01
#threshold = 1e-4


def compute_anomaly(results, params, seq_threshold=0.5):
    is_logkey = params["is_logkey"]
    total_errors = 0
    for seq_res in results:
        # label pairs as anomaly when over half of masked tokens are undetected
        if ((is_logkey and seq_res["undetected_tokens"]
             > seq_res["masked_tokens"] * seq_threshold) or
            (params["hypersphere_loss_test"] and seq_res["deepSVDD_label"])):
            total_errors += 1
        else:
            pass
            #if seq_res["undetected_tokens"] > 0:
            #    print('#' * 20)
            #    print('a: {}, b: {}'.format(seq_res["undetected_tokens"], seq_res["masked_tokens"]))
    #exit(0)
    return total_errors


def find_best_threshold(
    test_normal_results,
    test_abnormal_results,
    params,
):
    FP = compute_anomaly(test_normal_results, params, threshold)
    TP = compute_anomaly(test_abnormal_results, params, threshold)
    TN = len(test_normal_results) - FP
    FN = len(test_abnormal_results) - TP
    #print('TP: {}, FN: {}'.format(TP, FN))
    #exit(0)
    precision = 100 * TP / (TP + FP)
    recall = 100 * TP / len(test_abnormal_results)
    F1 = 2 * precision * recall / (precision + recall)
    return (FP, TP, TN, FN, precision, recall, F1)


class Predictor():

    def __init__(self, options):
        self.model_path = options["model_path"]
        self.vocab_path = options["vocab_path"]
        self.output_path = options["output_dir"]
        self.device = options["device"]
        self.window_size = options["window_size"]
        self.adaptive_window = options["adaptive_window"]
        self.seq_len = options["seq_len"]
        self.on_memory = options["on_memory"]
        self.batch_size = options["batch_size"]
        self.num_workers = options["num_workers"]
        self.num_candidates = options["num_candidates"]
        self.output_dir = options["output_dir"]
        self.model_dir = options["model_dir"]

        self.is_logkey = options["is_logkey"]
        self.is_param = options["is_param"]

        self.hypersphere_loss = options["hypersphere_loss"]
        #self.hypersphere_loss_test = options["hypersphere_loss_test"]
        self.hypersphere_loss_test = False

        self.center = None
        self.radius = None
        self.test_ratio = options["test_ratio"]
        self.mask_ratio = options["mask_ratio"]
        self.min_len = options["min_len"]
        self.st = SentenceTransformer('all-MiniLM-L6-v2')

    def detect_logkey_anomaly(self, masked_output, masked_label):
        num_undetected_tokens = 0
        output_maskes = []
        for i, token in enumerate(masked_label):
            if token not in torch.argsort(
                    -masked_output[i])[:self.num_candidates]:
                num_undetected_tokens += 1

        return num_undetected_tokens, [
            output_maskes, masked_label.cpu().numpy()
        ]

    @staticmethod
    def generate_test(output_dir, file_name, window_size, adaptive_window,
                      seq_len, min_len):
        log_seqs = []
        idx_seqs = []
        with open(
                os.path.join(output_dir,
                             '{}-{}'.format(file_name, "EventSequence"))) as f:
            with open(
                    os.path.join(output_dir,
                                 '{}-{}'.format(file_name,
                                                'LogSequence'))) as g:
                for idx, t in tqdm(enumerate(zip(f.readlines(),
                                                 g.readlines()))):
                    log_seq, idx_seq = fixed_window(
                        t,
                        window_size,
                        adaptive_window=adaptive_window,
                        seq_len=seq_len,
                        min_len=min_len)
                    if len(log_seq) == 0:
                        continue

                    log_seqs += log_seq
                    idx_seqs += idx_seq

        # sort seq_pairs by seq len
        log_seqs = np.asarray(log_seqs, dtype="object")
        idx_seqs = np.asarray(idx_seqs, dtype="object")

        test_len = list(map(len, log_seqs))
        test_sort_index = np.argsort(-1 * np.array(test_len))

        log_seqs = log_seqs[test_sort_index]
        idx_seqs = idx_seqs[test_sort_index]

        print(f"{file_name} size: {len(log_seqs)}")
        return log_seqs, idx_seqs

    def read_pickle(self, param_context_path):
        with open(os.path.join(self.output_path, param_context_path),
                  'rb') as f:
            return pickle.load(f)

    def helper(
        self,
        model,
        output_dir,
        file_name,
        vocab,
    ):
        total_results = []
        total_errors = []
        output_results = []
        total_dist = []
        logkey_test, log_seq_test = self.generate_test(output_dir, file_name,
                                                       self.window_size,
                                                       self.adaptive_window,
                                                       self.seq_len,
                                                       self.min_len)

        # use 1/10 test data
        if self.test_ratio != 1:
            num_test = len(logkey_test)
            rand_index = torch.randperm(num_test)
            rand_index = rand_index[:int(num_test *
                                         self.test_ratio)] if isinstance(
                                             self.test_ratio, float
                                         ) else rand_index[:self.test_ratio]
            logkey_test, log_seq_test = logkey_test[rand_index], log_seq_test[
                rand_index]

        param_context = self.read_pickle('context.pkl')
        seq_dataset = LogDataset(
            logkey_test,
            log_seq_test,
            vocab,
            seq_len=self.seq_len,
            on_memory=self.on_memory,
            predict_mode=True,
            mask_ratio=self.mask_ratio,
            param_context=param_context,
        )

        # use large batch size in test data
        data_loader = DataLoader(seq_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 collate_fn=seq_dataset.collate_fn)

        for idx, data in tqdm(enumerate(data_loader)):
            if True:
                st_x = []
                for idx_i in range(len(data["bert_input"])):
                    for idx_j in range(len(data["bert_input"][idx_i])):
                        if (data["bert_input"][idx_i][idx_j] != 0
                                and data["bert_input"][idx_i][idx_j] != 3):
                            assert data["context"][idx_i][idx_j] is not None
                            st_x.append(data["context"][idx_i][idx_j])
                        else:
                            assert data["context"][idx_i][idx_j] is None
                st_x = self.st.encode(
                    sentences=st_x,
                    output_value="sentence_embedding",
                    convert_to_numpy=True,
                )
                st_x = torch.nn.functional.adaptive_avg_pool1d(
                    torch.from_numpy(np.asarray(st_x)), 256).numpy()
                data["context"] = np.zeros((*data["bert_input"].shape, 256),
                                           dtype=np.float32)
                mask = (data["bert_input"] != 0) & (data["bert_input"] != 3)
                data["context"][mask] = st_x
                data["context"] = torch.from_numpy(data["context"])
            data = {key: value.to(self.device) for key, value in data.items()}
            result = model(data["bert_input"], data["context"])
            mask_lm_output = result["logkey_output"]

            # loop though each session in batch
            for i in range(len(data["bert_label"])):
                seq_results = {
                    "num_error": 0,
                    "undetected_tokens": 0,
                    "masked_tokens": 0,
                    "total_logkey":
                    torch.sum(data["bert_input"][i] > 0).item(),
                    "deepSVDD_label": 0
                }

                mask_index = data["bert_label"][i] > 0
                num_masked = torch.sum(mask_index).tolist()
                seq_results["masked_tokens"] = num_masked

                if self.is_logkey:
                    num_undetected, output_seq = self.detect_logkey_anomaly(
                        mask_lm_output[i][mask_index],
                        data["bert_label"][i][mask_index])
                    seq_results["undetected_tokens"] = num_undetected
                    output_results.append(output_seq)

                if self.hypersphere_loss_test:
                    # detect by deepSVDD distance
                    assert result["cls_output"][i].size() == self.center.size()
                    dist = torch.sqrt(
                        torch.sum((result["cls_output"][i] - self.center)**2))
                    total_dist.append(dist.item())

                    # user defined threshold for deepSVDD_label
                    seq_results["deepSVDD_label"] = int(
                        dist.item() > self.radius)
                total_results.append(seq_results)

        return total_results

    @torch.no_grad()
    def predict(self):
        model = torch.load(self.model_path, weights_only=False)
        model.to(self.device)
        model.eval()
        print('model_path: {}'.format(self.model_path))

        start_time = time.time()
        vocab = WordVocab.load_vocab(self.vocab_path)

        if self.hypersphere_loss:
            center_dict = torch.load(os.path.join(self.model_dir,
                                                  "best_center.pt"),
                                     weights_only=False)
            self.center = center_dict["center"]
            self.radius = center_dict["radius"]

        results = list()
        for x in [
                "test_normal",
                "test_abnormal",
        ]:
            results.append(self.helper(
                model,
                self.output_dir,
                x,
                vocab,
            ))

        params = {
            "is_logkey": self.is_logkey,
            "hypersphere_loss": self.hypersphere_loss,
            "hypersphere_loss_test": self.hypersphere_loss_test
        }
        FP, TP, TN, FN, precision, recall, F1 = find_best_threshold(
            results[0],
            results[1],
            #None,
            #results[0],
            params=params,
        )

        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        print('Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(
            precision, recall, F1))
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
