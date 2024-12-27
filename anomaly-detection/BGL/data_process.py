import sys

sys.path.append('../')

import os
from collections import defaultdict
import gc
import pandas as pd
import numpy as np
from logparser import Spell, Drain_BGL
import argparse
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

tqdm.pandas()
pd.options.mode.chained_assignment = None

data_dir = os.path.join('..', '..', '..', 'BGL')
output_dir = "../output/bgl/"
log_file = "BGL.log"


def fixed_len_window(raw_data, para):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=[timestamp, label, eventid, time duration]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe columns=[eventids, time durations, label]
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    logkey_data, deltaT_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3]
    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para['step_size']
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
            print("process {} time window".format(num_session), end='\r')
    # debug - lezhang.thu - start
    idx_seq = defaultdict(list)
    # debug - lezhang.thu - end
    for (start_index, end_index) in start_end_index_pair:
        dt = deltaT_data[start_index:end_index].values
        dt[0] = 0
        new_data.append([
            time_data[start_index:end_index].values,
            max(label_data[start_index:end_index]),
            logkey_data[start_index:end_index].values, dt
        ])
        # debug - lezhang.thu - start
        idx_seq['LogSequence'].append(list(range(start_index, end_index)))
        # debug - lezhang.thu - end

    assert len(start_end_index_pair) == len(new_data)
    print('there are %d instances (sliding windows) in this dataset\n' %
          len(start_end_index_pair))
    return pd.DataFrame(new_data,
                        columns=raw_data.columns), pd.DataFrame(idx_seq)


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+',  #hexadecimal
        r'\d+.\d+.\d+.\d+',
        # r'/\w+( )$'
        r'\d+'
    ]
    keep_para = True
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain_BGL.LogParser(log_format,
                                     indir=input_dir,
                                     outdir=output_dir,
                                     depth=depth,
                                     st=st,
                                     rex=regex,
                                     keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir,
                                 outdir=output_dir,
                                 log_format=log_format,
                                 tau=tau,
                                 rex=regex,
                                 keep_para=keep_para)
        parser.parse(log_file)


def mapping():
    import pickle
    with open(output_dir + log_file + "_templates.pkl", 'rb') as f:
        df = pickle.load(f)
    df.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    df_dict = {event: idx + 1 for idx, event in enumerate(list(df["EventId"]))}
    return df_dict


if __name__ == "__main__":
    ##########
    # Parser #
    #########

    parse_log(data_dir, output_dir, log_file, 'drain')

    ##################
    # Transformation #
    ##################
    # mins
    window_size = 5
    step_size = 1
    train_ratio = 0.6

    import pickle
    with open(f'{output_dir}{log_file}_structured.pkl', 'rb') as f:
        df = pickle.load(f)

    event_num = mapping()
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    # data preprocess
    df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10**9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)

    # sampling with fixed length window
    deeplog_df, idx_seq_df = fixed_len_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={
            "window_size": window_size * 60,
            "step_size": step_size * 60,
        })

    #########
    # Train #
    #########
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(
        drop=True)  #shuffle
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    # debug - lezhang.thu - start
    idx_seq_normal = idx_seq_df[deeplog_df["Label"] == 0]
    idx_seq_normal = idx_seq_normal.sample(
        frac=1, random_state=12).reset_index(drop=True)
    # debug - lezhang.thu - end

    train = df_normal[:train_len]
    test_normal = df_normal[train_len:]

    # debug - lezhang.thu - start
    train_idx = idx_seq_normal[:train_len]
    test_normal_idx = idx_seq_normal[train_len:]
    # debug - lezhang.thu - end

    deeplog_file_generator(
        os.path.join(output_dir, 'train-{}'.format("EventSequence")), train,
        ["EventId"])

    # debug - lezhang.thu - start
    deeplog_file_generator(
        os.path.join(output_dir, 'train-{}'.format("LogSequence")), train_idx,
        ["LogSequence"])
    # debug - lezhang.thu - end
    print("training size {}".format(len(train)))

    ###############
    # Test Normal #
    ###############
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal-{}".format("EventSequence")),
        test_normal, ["EventId"])
    # debug - lezhang.thu - start
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal-{}".format("LogSequence")),
        test_normal_idx, ["LogSequence"])
    # debug - lezhang.thu - end
    print("test normal size {}".format(len(test_normal)))

    del df_normal
    del train
    del test_normal
    gc.collect()

    #################
    # Test Abnormal #
    #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    deeplog_file_generator(
        os.path.join(output_dir, "test_abnormal-{}".format("EventSequence")),
        df_abnormal, ["EventId"])
    # debug - lezhang.thu - start
    df_abnormal_idx = idx_seq_df[deeplog_df["Label"] == 1]
    deeplog_file_generator(
        os.path.join(output_dir, "test_abnormal-{}".format("LogSequence")),
        df_abnormal_idx, ["LogSequence"])
    # debug - lezhang.thu - end
    print('test abnormal size {}'.format(len(df_abnormal)))
