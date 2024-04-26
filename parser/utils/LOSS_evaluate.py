import math
import pandas as pd
import re
def loss(df_t: pd.DataFrame):
    QL = 0
    for idx, line in df_t.iterrows():
        template = line['EventTemplate']
        template_seq = str.split(template)

        parameter_count = 0

        for i in range(len(template_seq)-1 , -1, -1):
            token = template_seq[i]
            if len(token) == 1 and token[0] in ':,=$@#;_|':
                del template_seq[i]

        for token in template_seq:
            if str.find(token, '<*>') >= 0:
                parameter_count += 1

        QL += (parameter_count / len(template_seq)) ** 2

        #print('%.3f'%(parameter_count / len(template_seq)) ** 2, parameter_count, template)
        
    print('QL=', QL)
    LL = math.log(len(df_t))**1.5
    print('LL=', LL)
    print('LOSS=', QL + LL)
    return QL, LL
