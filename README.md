# LogBERT

## Runnable version

1. Download HDFS_v1 from [https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1](https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1)
2. Run `unzip -q HDFS_v1.zip`
3. Run the following scripts
```[bash]
cd anomaly-detection/HDFS
python data_process /path/to/HDFS-dataset
python logser.py vocab
python logser.py train
python logser.py predict
```
4. We use the default log parser as [LogBert](https://github.com/HelenGuohx/logbert/blob/main/HDFS/data_process.py#L119), i.e., 'drain'.
