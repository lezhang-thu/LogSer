import sys

sys.path.append('../')

import os
from collections import defaultdict
import pandas as pd
import numpy as np
from logparser import Spell, Drain
from tqdm import tqdm
import json

tqdm.pandas()
pd.options.mode.chained_assignment = None  # default='warn'

data_dir = os.path.join('..', '..', '..', 'tbird')
output_dir = "../output/tbird/"
log_file = "Thunderbird_20M.log"


def sliding_window(raw_data, para):
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
    log_format = '<Label> <Id> <Date> <Admin> <Month> <Day> <Time> <AdminAddr> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+',  # hexadecimal
        r'\d+\.\d+\.\d+\.\d+',
        r'(?<=Warning: we failed to resolve data source name )[\w\s]+',
        r'\d+'
    ]
    keep_para = True
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes

        # Drain is modified
        parser = Drain.LogParser(log_format,
                                 indir=input_dir,
                                 outdir=output_dir,
                                 depth=depth,
                                 st=st,
                                 rex=regex,
                                 keep_para=keep_para,
                                 maxChild=1000)
        parser.parse(log_file)

    elif parser_type == "spell":
        tau = 0.35
        parser = Spell.LogParser(indir=data_dir,
                                 outdir=output_dir,
                                 log_format=log_format,
                                 tau=tau,
                                 rex=regex,
                                 keep_para=keep_para)
        parser.parse(log_file)


def sample_raw_data(data_file, output_file, sample_window_size,
                    sample_step_size):
    # sample 1M by sliding window, abnormal rate is over 2%
    sample_data = []
    labels = []
    idx = 0

    # spirit dataset can start from the 2Mth line, as there are many abnormal lines gathering in the first 2M
    with open(data_file, 'r', errors='ignore') as f:
        for line in f:
            labels.append(line.split()[0] != '-')
            sample_data.append(line)

            if len(labels) == sample_window_size:
                abnormal_rate = sum(np.array(labels)) / len(labels)
                print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                break

            idx += 1
            if idx % sample_step_size == 0:
                print("Process {:>6.2f}% raw data".format(
                    idx / sample_window_size * 100),
                      end='\r')

    with open(output_file, "w") as f:
        f.writelines(sample_data)

    print("Sampling done")


log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"


def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {
        event: idx + 1
        for idx, event in enumerate(list(log_temp["EventId"]))
    }
    with open(output_dir + "tbird_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


if __name__ == "__main__":
    print("output_dir:", output_dir)
    raw_log_file = "Thunderbird.log"
    sample_log_file = "Thunderbird_20M.log"
    sample_window_size = 2 * 10**7
    sample_step_size = 10**4
    window_name = ''
    log_file = sample_log_file

    parser_type = 'drain'
    #mins
    window_size = 1
    step_size = 0.5
    train_ratio = 6000

    #########
    # sample raw data
    #########
    if False:
        sample_raw_data(os.path.join(data_dir, raw_log_file),
                        os.path.join(data_dir, sample_log_file),
                        sample_window_size, sample_step_size)

    ##########
    # Parser #
    #########
    if True:
        parse_log(data_dir, output_dir, log_file, 'drain')

    mapping()
    ##################
    # Transformation #

    ##################
    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')
    with open(output_dir + "tbird_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    # data preprocess
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))

    df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'],
                                    format='%Y.%m.%d %H:%M:%S')
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10**9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)

    # sampling with sliding window
    deeplog_df, idx_seq_df = sliding_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={
            "window_size": float(window_size) * 60,
            "step_size": float(step_size) * 60
        })

    #########
    # Train #
    #########
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(
        drop=True)  #shuffle
    normal_len = len(df_normal)
    train_len = int(train_ratio) if train_ratio >= 1 else int(normal_len *
                                                              train_ratio)

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
    print("training size {}".format(train_len))

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
    print("test normal size {}".format(normal_len - train_len))

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
