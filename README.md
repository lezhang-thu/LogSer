# LogSer

Log Sequence Anomaly Detection based on Template and Parameter Parsing via BERT

## Runnable version

1. Download HDFS_v1 from [https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1)

1. Extract it in **the same level** of directory as LogSer (i.e., put it in the same level as LogSer, then run `unzip -q HDFS_v1.zip`)

1. We use the default log parser as [LogBert](https://github.com/HelenGuohx/logbert/blob/main/HDFS/data_process.py#L119), i.e., 'drain'. We uncomment `parser(input_dir, output_dir, log_file, log_format, 'drain')` in `anomaly detection/HDFS/data_process.py` to achieve it.

The current version NOT supports LogSer's parser, but uses drain instead.

**Notes:** In LogSer `parser/LogSer_benchmark.py`, there exists `'blk_-?\d+':'(BLK)'`. It might explain why `blkId_list` in `anomaly detection/HDFS/data_process.py` is always `[]`.
