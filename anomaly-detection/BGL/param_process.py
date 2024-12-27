import sys

sys.path.append('../')

from logparser import bgl_param_transformer
import json
from collections import defaultdict
import os

label_dir = None
output_dir = '../output/bgl/'  # The output directory of parsing results
log_file = "BGL.log"  # The input log file name


def param_embedding():
    model_trans = bgl_param_transformer.ParamTransformer(
        logName=log_file,
        label_dir=label_dir,
        indir=output_dir,
        outdir=output_dir,
        modelname="all-MiniLM-L6-v2")
    model_trans.model_embedding()


if __name__ == "__main__":
    # generate parameter embedding
    param_embedding()
