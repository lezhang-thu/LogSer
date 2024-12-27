import sys

sys.path.append('../')

from logparser import tbird_param_transformer
import json
from collections import defaultdict
import os

label_dir = None
output_dir = '../output/tbird/'  # The output directory of parsing results
log_file = "Thunderbird_20M.log"  # The input log file name


def param_embedding():
    model_trans = tbird_param_transformer.ParamTransformer(
        logName=log_file,
        label_dir=label_dir,
        indir=output_dir,
        outdir=output_dir,
        modelname="all-MiniLM-L6-v2")
    model_trans.model_embedding()


if __name__ == "__main__":
    # generate parameter embedding
    param_embedding()
