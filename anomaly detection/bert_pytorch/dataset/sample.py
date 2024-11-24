from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def fixed_window(t, window_size, adaptive_window, seq_len=None, min_len=0):
    line = [x.split(",") for x in t[0].split()]
    idx_seq = [x.split(",") for x in t[1].split()]

    # filter the line/session shorter than 10
    if len(line) < min_len:
        return [], []

    # max seq len
    if seq_len is not None:
        line = line[:seq_len]
        idx_seq = idx_seq[:seq_len]

    if adaptive_window:
        window_size = len(line)

    line = np.asarray(line)
    idx_seq = np.asarray(idx_seq)

    assert line.shape[1] == 1
    line = line.squeeze()
    idx_seq = idx_seq.squeeze()

    template_seq = []
    log_seq = []
    for i in range(0, len(line), window_size):
        template_seq.append(line[i:i + window_size])
        log_seq.append(idx_seq[i:i + window_size])

    return template_seq, log_seq


def generate_train_valid(data_path,
                         window_size=20,
                         adaptive_window=True,
                         sample_ratio=1,
                         valid_size=0.1,
                         output_path=None,
                         scale=None,
                         scale_path=None,
                         seq_len=None,
                         min_len=0):
    with open(data_path[0], 'r') as f:
        data_iter = f.readlines()
    with open(data_path[1], 'r') as f:
        idx_seq = f.readlines()
    assert len(data_iter) == len(idx_seq)

    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    print("=" * 40)

    logkey_seq_pairs = []
    idx_seq_pairs = []
    session = 0
    for line in tqdm(zip(data_iter, idx_seq)):
        #for line in data_iter:
        if session >= num_session:
            break
        session += 1

        logkeys, idx_seqs = fixed_window(line, window_size, adaptive_window,
                                         seq_len, min_len)
        #print('#' * 20)
        #print('logkeys: {}'.format(logkeys))
        #print('len(logkeys[0]): {}'.format(len(logkeys[0])))
        #exit(0)
        logkey_seq_pairs += logkeys
        idx_seq_pairs += idx_seqs

    logkey_seq_pairs = np.asarray(logkey_seq_pairs, dtype="object")
    idx_seq_pairs = np.asarray(idx_seq_pairs, dtype="object")

    logkey_trainset, logkey_validset, idx_seq_trainset, idx_seq_validset = train_test_split(
        logkey_seq_pairs, idx_seq_pairs, test_size=test_size, random_state=1234)

    # sort seq_pairs by seq len
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    idx_seq_trainset = idx_seq_trainset[train_sort_index]
    idx_seq_validset = idx_seq_validset[valid_sort_index]

    print("=" * 40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("=" * 40)

    return logkey_trainset, logkey_validset, idx_seq_trainset, idx_seq_validset
