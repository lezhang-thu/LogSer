# -*- coding: UTF-8 -*-
import re
import os
import sys
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime
import math

import io


class Logcluster:
    def __init__(self, logTemplate:list[str]=[], logIDL=None):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL
        self.levelL = [level(token) for token in logTemplate]

    def getDist(self, logSeq:list[str], logSeqLevel:list[int]):
        assert len(self.logTemplate) == len(logSeq)
        simlarity = 0.0
        sameTokens = 0

        for token1, token2, level, log_level in zip(self.logTemplate, logSeq, self.levelL, logSeqLevel):
            if token1 == token2:
                simlarity += 1 if token1 != '<*>' else 0.5
                sameTokens += 1
            elif token1 == '<*>':
                simlarity += 0.5 if log_level == 2 else 0
            else:
                simlarity += 0.5 if level == log_level == 2 else 0

        return simlarity / len(self.logTemplate) if sameTokens != len(self.logTemplate) else 1

    def appendLog(self, logSeq:list[str], logID:int):
        assert len(self.logTemplate) == len(logSeq)
        self.logIDL.append(logID)
        for i in range(len(logSeq)):
            if logSeq[i] != self.logTemplate[i]:
                self.logTemplate[i] = '<*>'
                self.levelL[i] = 1

class Node:
    def __init__(self, depth:int, digitOrToken):
        self.childD = dict()
        self.clusters = list()
        self.depth = depth
        self.digitOrToken = digitOrToken

def level(s:str):
    '''measure the posibility that the token is a parameter'''
    l = 0
    for char in s:
        if char.isdigit():
            return 2
        elif char in '#^$\'*+,/<=>@_)|~':
            l = 1
    return l

template_regex_map = {}
class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, ht=0.4, 
                 maxChild=100, rex=[], keep_para=True, jt = 1.0, postProcessFunc = None, replaceD = {}):
        self.path = indir
        self.depth = depth - 2
        self.ht = ht
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.jt = jt
        self.mergeFunc = postProcessFunc
        self.replaceD = replaceD

    def treeSearch(self, rn:Node, seq:list[str]):
        seqLen = len(seq)
        if seqLen not in rn.childD:
            return None

        curNode:Node = rn.childD[seqLen]
        curDepth = 1
        i = 0
        j = len(seq) - 1
        seqLevel = [level(token) for token in seq]
        while i <= j and i < seqLen and j >= 0:
            
            if curDepth >= self.depth:
                break
            
            while i < seqLen and seqLevel[i] == 2:
                i += 1
            while j >= 0 and seqLevel[j] == 2:
                j -= 1
            if i > j or i > seqLen or j < 0:
                break

            if seqLevel[i] <= seqLevel[j]:
                token = seq[i]
                i += 1
            else:
                token = seq[j]
                j -= 1

            if token in curNode.childD:
                curNode = curNode.childD[token]
            elif '<*>' in curNode.childD:
                curNode = curNode.childD['<*>']
            else:
                return None
            curDepth += 1

        return self.fastMatch(curNode.clusters, seq, seqLevel)

    def addSeqToPrefixTree(self, rn:Node, logClust:Logcluster):

        seqLen = len(logClust.logTemplate)
        seq = logClust.logTemplate
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrToken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        curNode = firtLayerNode

        curDepth = 1
        i = 0
        j = seqLen - 1
        while i <= j and i < seqLen and j >= 0:
            if curDepth >= self.depth:
                break

            while i < seqLen and level(seq[i]) == 2:
                i += 1
            while j >= 0 and level(seq[j]) == 2:
                j -= 1
            if i > j:
                break

            if level(seq[i]) <= level(seq[j]):
                token = seq[i]
                i += 1
            else:
                token = seq[j]
                j -= 1

            if token not in curNode.childD:
                if '<*>' in curNode.childD:
                    if len(curNode.childD) < self.maxChild:
                        newNode = Node(depth=curDepth + 1, digitOrToken=token)
                        curNode.childD[token] = newNode
                        curNode = newNode
                    else:
                        curNode = curNode.childD['<*>']
                else:
                    
                    if len(curNode.childD)+1 < self.maxChild:
                        newNode = Node(depth=curDepth+1, digitOrToken=token)
                        curNode.childD[token] = newNode
                        curNode = newNode
                    elif len(curNode.childD)+1 == self.maxChild:
                        newNode = Node(depth=curDepth+1, digitOrToken='<*>')
                        curNode.childD['<*>'] = newNode
                        curNode = newNode
                    else:
                        curNode = curNode.childD['<*>']
            else:
                curNode = curNode.childD[token]
            curDepth += 1

        curNode.clusters.append(logClust)

    def fastMatch(self, logClustL:list[Logcluster], seq:list[str], seqLevel:list[int]):
        if len(logClustL) == 0:
            return None
        bestCluster = max(logClustL, key=lambda l: l.getDist(seq, seqLevel))
        return bestCluster if bestCluster.getDist(seq, seqLevel) >= self.ht else None

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        for logClust in logClustL:
            #print(logClust.logTemplate)
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])
        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        print('extracting parameters...')
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
            self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)
        else:
            self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.sort_values(by='Occurrences', inplace=True)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        print('HammingThreshold=%.2f'%self.ht)
        print('JaccardThreshold=%.2f'%self.jt)
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node(depth=0, digitOrToken=None)
        logCluL = []

        self.load_data()

        count = 0
        paraPattern = re.compile(r'^-?\d+(\.\d+)?$|^(0x)?([0-9a-fA-F]{4})+$')
        for idx, line in self.df_log.iterrows():
            #print(line['LineId'], line['Content'])
            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            for i in range(len(logmessageL)):
                if paraPattern.match(logmessageL[i]) is not None:
                    logmessageL[i] = '<*>'
                else:
                    logmessageL[i] = logmessageL[i]
                logmessageL[i] = '<*>' if re.match(r'^-?\d+(\.\d+)?$', logmessageL[i]) is not None else logmessageL[i] 
            matchCluster = self.treeSearch(rootNode, logmessageL)

            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)
            else:
                matchCluster.appendLog(logmessageL, logID)

            count += 1
            if count % 10000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        if not self.mergeFunc == None:
            logCluL = mergeClusters(logCluL, self.jt, self.mergeFunc)


        self.outputResult(logCluL)
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))


    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def get_parameter_list(self, row):
        template = template_regex = row["EventTemplate"]
        if "<*>" not in template_regex: return []
        if template in template_regex_map.keys(): 
            template_regex = template_regex_map[template]
        else:
            template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
            template_regex = re.sub(r'\\ +', ' ?', template_regex)
            template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
            template_regex_map[template] = template_regex
        parameter_list = re.findall(template_regex, re.sub(' +', ' ', row["Content"]))
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return list(filter(None, parameter_list))
    
    def log_to_dataframe(self, log_file, regex, headers:list, logformat):
        log_messages = []
        linecount = 0
        content_index = headers.index('Content')
        with open(log_file, 'r', encoding='utf8') as fin:
            lines = fin.readlines()
            for line in lines:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    message[content_index] = self.perprocess1(message[content_index])
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
                if linecount % 100000 == 0:
                    print(linecount, ' lines loaded')
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        print(linecount, 'lines loaded')
        return logdf

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    
    def perprocess1(self, content):
        for key, value in self.replaceD.items():
            content = re.sub(key, value, content)
        return re.sub(' +',' ',content)    

def mergeClusters(logCluL:list, threshold:float, mergeFunc):
    templateCluL = []
    for logCluster in logCluL:
        isMerged = False
        for templateCluster in templateCluL:
            if templateCluster.tryMerge(logCluster, threshold):
                isMerged = True
                break
        if not isMerged:
            templateCluL.append(TemplateCluster(logCluster, mergeFunc))
    newLogClulL = []
    for templateCluster in templateCluL:
        newLogClulL.append(templateCluster.generateLogCluster())
    return newLogClulL

class TemplateCluster:
    def __init__(self, cluster: Logcluster, mergeFunc):
        self.logTemplate = cluster.logTemplate
        self.logClusterL = [cluster]
        self.lenL = [len(cluster.logTemplate)]
        self.mergeFunc = mergeFunc

    def tryMerge(self, newCluster: Logcluster, threshold: float)->bool:
        
        check, newTemplate = self.mergeFunc(newCluster.logTemplate, self.logTemplate, threshold)
        if not check:
            return False
        self.logTemplate = newTemplate
        self.logClusterL.append(newCluster)
        #self.lenL.append(len(newCluster.logTemplate))
        return True
        
    def generateLogCluster(self)->Logcluster:
        logIDL = []
        for logCluster in self.logClusterL:
            logIDL.extend(logCluster.logIDL)
        return Logcluster(self.logTemplate, logIDL)