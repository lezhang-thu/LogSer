import math
import pandas as pd
import os
import re
def compare(groundtruth:list[str], parsedresult:list[str], parsed_occurence:list[int]) -> float:
    count = 0
    for i in range(len(parsedresult)):
        if parsedresult[i] in groundtruth:
            count += parsed_occurence[i]
    return count / sum(parsed_occurence)

allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*<>')
def preprocess(t:str)->str:
    t=re.sub(r'\(BLK\)', '<*>', t)
    t=re.sub(r'\(ADDR\)', '<*>', t)
    t=re.sub(r'\(CORE\)', '<*>', t)
    t=re.sub(r'\(HWID\)', 'HWID=<*>', t)
    t=re.sub(r'\(PID\)', '<*>', t)
    t=re.sub(r'\(ADDR\)', '<*>', t)
    t=re.sub(r'\(CHILD\)', 'child <*>', t)
    t=re.sub(r'\(INST\)', '<*>', t)
    t=re.sub(r'\(APRT\)', 'APRT: <*>', t)
    return ''.join(c for c in t if c in allowed_chars)

def evaluate(groundtruth_path:str, parsedresult_path:str, show_details=False):
    try:
        groundtruth:list[str] = pd.read_csv(groundtruth_path)['EventTemplate'].tolist()
        parsedresult:list[str] = pd.read_csv(parsedresult_path)['EventTemplate'].tolist()
        groundtruth = [preprocess(t) for t in groundtruth]
        parsedresult = [preprocess(t) for t in parsedresult]
        parsed_occurence:list[int] = pd.read_csv(parsedresult_path)['Occurrences'].tolist()
        if show_details:
            print('ground truth')
            with open('groundtruth', 'w') as f:
                f.writelines([line+'\n' for line in sorted(groundtruth)])
            for s in sorted(groundtruth):
                print(s)
            print('\nparsed result')
            with open('parsedresult', 'w') as f:
                f.writelines([line+'\n' for line in sorted(parsedresult)])
            for s in sorted(parsedresult):
                print(s)
    except ZeroDivisionError:
        return 0
    except Exception as e:
        return math.nan
    return compare(groundtruth, parsedresult, parsed_occurence)


result_dirs = {
    'logser': r'E:\日志解析-大修\LogSerPlus\LogSer_results',
}
groundtruth_dir = r'E:\日志解析-大修\logparser\data\loghub_2k_corrected'
datasets = ['Android', 'Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux', 'Mac', 
            'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Windows', 'Zookeeper']

def evaluate_all():

    models = ['logser']
    df = pd.DataFrame(columns=['dataset'] + models)

    for dataset in datasets:
        temp_dict = {'dataset': dataset}
        for model in models:
            result = evaluate(os.path.join(groundtruth_dir, dataset, f'{dataset}_2k.log_templates_corrected.csv'), os.path.join(result_dirs[model], f'{dataset}_2k.log_templates.csv'))
            temp_dict[model] = result

        df.loc[len(df)] = temp_dict

    average_dict = {'dataset': 'average'}
    for model in models:
        average_dict[model] = df[model].mean()
    df.loc[len(df)] = average_dict

    df.to_csv('PA_2kresult.csv')
    print(df)

def evaluate_one(dataset):
    evaluate(os.path.join(groundtruth_dir, dataset, f'{dataset}_2k.log_templates_corrected.csv'), os.path.join(result_dirs['logser'], f'{dataset}_2k.log_templates.csv'), show_details=True)

if __name__ == '__main__':
    evaluate_all()

