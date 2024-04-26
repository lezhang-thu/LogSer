#!/usr/bin/env python
import os

import pandas as pd

from LogSer import LogSer
from LogSer.Jaccard import Jaccard, LCS
from utils import evaluator
from utils import LOSS_evaluate
from datetime import datetime


benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_full.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+(:\d+)?'],
        'st': 0.7,
        'tau': 0.8,
        'depth': 3,
        'replaceD': {
            'blk_-?\d+':'(BLK)',
            #':':' : '
        }
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_full.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.6,
        'tau': 0.7,
        'depth': 3,      
        'replaceD': {
            ':':' : ',
            #'$':' ',
            #'_':' ',
            '@':' @ '
        }
        },

    'Spark': {
        'log_file': 'Spark/Spark_full.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.7,
        'tau': 0.8,
        'depth': 3,
        'replaceD': {
            #'_':' _ '
            r'\(':'( ',
            r'\)':' )'
        }
        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_full.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'st': 0.8,
        'tau': 0.8,
        'depth': 3,
        'replaceD': {
            ':':' : ',
            '$':' $ '
        }        
        },

    'BGL': {
        'log_file': 'BGL/BGL_full.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+', r'(?<=\.{2})((0x[0-9A-Fa-f]+)|(\d+))', ],
        'st': 0.74,
        'tau': 1.0,
        'depth': 3,        
        'replaceD': {
            ',':' , ',
            #':':' : ',
            '=':' = ',
            r'\.\.\.\d+':'...<*>',
            r'\(':'( ',
            r'\)':' )',
            r'0x(\w){8}':'(ADDR)',
            r'core\.\d+':'(CORE)',
        }
        },

    'HPC': {
        'log_file': 'HPC/HPC_full.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'st': 0.7,
        'tau': 0.7,
        'depth': 3, 
        'replaceD': {
            r'HWID=\d+':'(HWID)',
            ':':' : ',
            #'=':' = '
        }
        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_full.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.7,
        'tau': 0.8,
        'depth': 4,       
        'replaceD': {
            r'\b([0-9a-zA-Z]){14}\b':'(PID)',
            #'#':' ',
            '=':' = ',
            r'\(': '( ',
            r'\)': ' )',
            r'[a-z]n\d+': '<*>',
            #r'(?<=[a-zA-Z]):(?=\d)':': '
        }
        },


    'Linux': {
        'log_file': 'Linux/Linux_full.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'st': 0.7,
        'tau': 0.6,
        'depth': 3,
        'replaceD': {
            #':':' : ',
            ';':'; ',
            ',':', ',
            #'_':' ',
            '\(':' ( ',
            '\)':' ) ',
            '=': ' = ',
            r'<': '< ',
            r'>': ' >',
        }       
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_full.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'st': 0.7,
        'tau': 0.6,
        'depth': 3,
        'replaceD': {
            '=': ' = ',
            #'##':' ',
            ':': ' : '
        }
        },

    'Apache': {
        'log_file': 'Apache/Apache_full.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'st': 0.5,
        'tau': 0.7,
        'depth': 3,
        'replaceD': {
            r'child \d+':'(CHILD)'
        }        
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_full.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'\d{2}:\d{2}(:\d{2})*', r'[0-9]+(\.[0-9]+)? [KGTM]B', r"([\w-]+\.)+[\w-]+(:\d+)?"],
        'st': 0.60,
        'tau': 1,
        'depth': 3,
        'replaceD': {
        }
        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_full.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.7,
        'tau': 1,
        'depth': 3,
        'replaceD': {
            ':':' : ',
            #'_':' ',
            '=':' = ',
            '\[': '[ ',
            '\]': ' ]'
        }   
        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_full.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/[^\s]*[a-zA-Z][^\s]*', r'\s(\/[a-zA-Z0-9_]+)+', r'([a-zA-Z0-9_-]{2,})(\.[a-zA-Z0-9_-]{2,})+', r'HTTP/1.1'],
        'st': 0.9,
        'tau': 0.85,
        'depth': 3,
        'replaceD': {
            r'\(': '( ',
            r'\)': ' )',
            r'[0-9a-z]{8}(-[0-9a-z]{4}){3}-[0-9a-z]{12}':'(INST)',
            #',':' ',
            #':':': ',
            #'=':' = '
        }   
        },

    'Mac': {
        'log_file': 'Mac/Mac_full.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'st': 0.8,
        'tau': 0.7,
        'depth': 3,
        'replaceD': {
            ',': ' , ',
            r'\(': '( ',
            r'\)': ' )',
            r'\[' : '[ ',
            r'\]' : ' ]',
            r'=(?!>)' : ' = ',
            r'\{' : '{ ',
            r'\}' : ' }',
            r'ARPT: \d+.\d+':'(APRT)',
        }     
        },
}
input_dir = 'PATH TO YOUR DATASETS'
output_dir = 'LogSer_bigdataset_results'
bechmark_result = []
if __name__ == '__main__':
    import sys
    if sys.argv.__len__() == 2:
        dataset = sys.argv[1]
        benchmark_settings = {dataset:benchmark_settings[dataset]}
    for dataset in benchmark_settings.keys():
        print('\n=== Evaluation on %s ==='%dataset)
        parser = LogSer.LogParser(log_format=benchmark_settings[dataset]['log_format'], 
                         indir=os.path.join(input_dir, os.path.dirname(benchmark_settings[dataset]['log_file'])), 
                         outdir=output_dir,  
                         depth=benchmark_settings[dataset]['depth'], 
                         ht=benchmark_settings[dataset]['st'], 
                         rex=benchmark_settings[dataset]['regex'], 
                         jt=benchmark_settings[dataset]['tau'], 
                         postProcessFunc = Jaccard, 
                         replaceD=benchmark_settings[dataset]['replaceD'],
                         keep_para=False)
        start_time = datetime.now()
        parser.parse(os.path.basename(benchmark_settings[dataset]['log_file']))
        parse_time = (datetime.now() - start_time).total_seconds()
        Precision, Recall, F1_measure, accuracy = evaluator.evaluate(
                    groundtruth=os.path.join(input_dir, benchmark_settings[dataset]['log_file'] + '_structured.csv'),
                    parsedresult=os.path.join(output_dir, dataset + '_full.log' + '_structured.csv')
                    )    
        QL, LL = LOSS_evaluate.loss(pd.read_csv(os.path.join(output_dir, dataset + '_full.log' + '_templates.csv')))
        bechmark_result.append([dataset, Precision, Recall, F1_measure, accuracy, parse_time, QL, LL, (QL+LL)])
        print([dataset, Precision, Recall, F1_measure, accuracy, parse_time, QL, LL, (QL+LL)])
        print('')

    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'Precision', 'Recall', 'F1_measure', 'Accuracy', 'Time', 'QL', 'LL', 'LOSS'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.to_csv('LogSer_bigdata_result.csv')