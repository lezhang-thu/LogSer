import os
from LogSer import LogSer
from LogSer.Jaccard import Jaccard
from LogSer_benchmark import benchmark_settings
from datetime import datetime
from utils import evaluator

def parse():
    dataset = 'OpenSSH'
    output_dir = 'LogSer_results'
    parser = LogSer.LogParser(log_format=benchmark_settings[dataset]['log_format'], 
                        indir="path to your datasets", 
                        outdir=output_dir,  
                        depth=benchmark_settings[dataset]['depth'], 
                        ht=0.6, 
                        rex=benchmark_settings[dataset]['regex'], 
                        jt=1, 
                        postProcessFunc = Jaccard, 
                        replaceD={
                            ':':' : ',
                            '_':' ',
                            '=':' = ',
                            '\[':'[ ',
                            '\]':' ]'
                        })
    start_time = datetime.now()
    parser.parse('OpenSSH_2k.log')
    print(f'parse time = {(datetime.now() - start_time).total_seconds()}')

def evaluate():
    print(evaluator.evaluate(r'logparser\data\loghub_2k_corrected\OpenSSH\OpenSSH_2k.log_structured_corrected.csv', r'LogSer_results\OpenSSH_2k.log_structured.csv'))

if __name__ == '__main__':
    parse()
    evaluate()