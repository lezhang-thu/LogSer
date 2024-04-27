import sys
sys.path.append('../')

from logparser import hdfs_param_transformer
import json
from collections import defaultdict
import os

label_dir  = os.path.expanduser('~/.dataset/hdfs/')
output_dir = '../output/hdfs/'  # The output directory of parsing results
log_file   = "HDFS.log"  # The input log file name

def param_embedding():
    model_trans = hdfs_param_transformer.ParamTransformer(
        logName=log_file,
        label_dir = label_dir, 
        indir=output_dir, 
        outdir=output_dir,
        modelname="all-MiniLM-L6-v2"
    )
    with open(output_dir + "hdfs_log_templates.json", "r") as f:
        event_num = json.load(f)

    model_trans.model_embedding()
    p_data_nor = model_trans.read_from_file(datatype="_embeddings_normal")    
    p_embeddings_nor = defaultdict()
    for eventid, embedding in p_data_nor.items():
        p_embeddings_nor[event_num.get(eventid, -1)] = embedding
    model_trans.write_to_file(p_embeddings_nor, datatype="_embeddings_normal")
    del p_embeddings_nor
    del p_data_nor

    p_data_abn = model_trans.read_from_file(datatype="_embeddings_abnormal")
    p_embeddings_abn = defaultdict()
    for eventid, embedding in p_data_abn.items():
        p_embeddings_abn[event_num.get(eventid, -1)] = embedding  
    model_trans.write_to_file(p_embeddings_abn, datatype="_embeddings_abnormal")
    del p_data_abn
    del p_embeddings_abn

    t_and_p_data_nor = model_trans.read_from_file(datatype="_with_template_embeddings_normal")
    t_and_p_embeddings_nor = defaultdict()
    for eventid, embedding in t_and_p_data_nor.items(): 
        t_and_p_embeddings_nor[event_num.get(eventid, -1)] = embedding
    model_trans.write_to_file(t_and_p_embeddings_nor, datatype="_with_template_embeddings_normal")
    del t_and_p_data_nor
    del t_and_p_embeddings_nor

    t_and_p_data_abn = model_trans.read_from_file(datatype="_with_template_embeddings_abnormal")
    t_and_p_embeddings_abn = defaultdict()
    for eventid, embedding in t_and_p_data_abn.items(): 
        t_and_p_embeddings_abn[event_num.get(eventid, -1)] = embedding
    model_trans.write_to_file(t_and_p_embeddings_abn, datatype="_with_template_embeddings_abnormal")

    print("param embedding done")

if __name__ == "__main__":
    # generate parameter embedding
    param_embedding()