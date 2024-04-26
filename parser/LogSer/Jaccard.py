# -*- coding: UTF-8 -*-
from LogSer.LCS import LCS
def Jaccard(seq1:list, seq2:list, t:float)->tuple:
    set1 = set(seq1)
    set2 = set(seq2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not len(intersection)*1.0/len(union) > t:
        return False, None
    return True, getTemplate(seq1, seq2)

def getTemplate(seq1:list, seq2:list)->list:
    b, retVal = LCS(seq1, seq2, 0)
    return retVal    

