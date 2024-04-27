
## Configuration
- Ubuntu 20.04.4 LTS
- NVIDIA driver 525.89.02 
- CUDA 12.0  
- Python 3.8.10
- torch 1.13.1 

## Installation
This code requires the packages listed in requirements.txt.
On Linux:  
```
pip install -r ./environment/requirements.txt
```

## Experiment
Our models are implemented on logPAI loghub including [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [BGL](https://github.com/logpai/loghub/tree/master/BGL), and [thunderbird](https://github.com/logpai/loghub/tree/master/Thunderbird) datasets

### HDFS example
First: initialing environment and downloading HDFS.log file
```shell script
cd HDFS

sh init.sh
```

Second: processing data
```shell script
# log parsing and sampling data
python data_process.py
# process parameter
python param_process.py
```

Finally: running model
```shell script
python logser.py vocab
python logser.py train
python logser.py predict
```

Running baselines
```shell script
#run deeplog
python deeplog.py vocab
# set options["vocab_size"] = <vocab output> above
python deeplog.py train
python deeplog.py predict 

#run loganomaly
python loganomaly.py vocab
# set options["vocab_size"] = <vocab output> above
python loganomaly.py train
python loganomaly.py predict

baselines.ipynb
```

