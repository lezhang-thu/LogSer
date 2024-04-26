## Get Start
1. Run benchmark
   - Download datasets from https://github.com/logpai/logparser/tree/main/data.
   - Change the input_dir in LogSer_benchmark.py to the path of your dataset.
   - Run `python LogSer_benchmark.py` to reproduce data on 2k scale datasets.
   - Run `cd utils && python PAevaluate.py` and `cd utils && python FTAevaluate.py` to obtain PA and FTA for the parsing results. You need to set the path for parsed result and ground truth in PAevaluate.py and FTAevaluate.py.
2. Running on big datasets
   - Download datasets from https://zenodo.org/records/8275861.
   - Change the input_dir in BigDataTest.py to the path of your dataset.
   - Run `python BigDataTest.py` to reproduce data on big datasets.