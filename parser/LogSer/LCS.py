# -*- coding: UTF-8 -*-
def LCS(seq1:list, seq2:list, t:float)->tuple:
    lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
    # row 0 and column 0 are initialized to 0 already
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if seq1[i] == seq2[j]:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    # read the substring out from the matrix
    if lengths[len(seq1)][len(seq2)] < max(len(seq1), len(seq2)) * t:
        return False, None
    result = []
    lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
    while lenOfSeq1!=0 and lenOfSeq2 != 0:
        if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
            lenOfSeq1 -= 1
        elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
            lenOfSeq2 -= 1
        else:
            assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
            result.insert(0,seq1[lenOfSeq1-1])
            lenOfSeq1 -= 1
            lenOfSeq2 -= 1
    return True, getTemplate(result, seq1, seq2)

def getTemplate(lcs:list, seq1:list, seq2:list)->list:
    retVal = []
    if not lcs:
        return retVal
    i = j = 0
    for token in lcs:
        #若i,j位置的词与lcs中的token相同，则将该词放入retVal
        if token == seq1[i] and token == seq2[j]:
            retVal.append(token)
            i = i + 1
            j = j + 1
            continue
        #若i,j位置的词都不与lcs中的token相同，则在retVal中放入<*>
        while token != seq1[i] and token != seq2[j]:
            retVal.append('<*>')
            i = i + 1
            j = j + 1
        #若i,j位置的词有一个不与lcs的token相同，则添加一个<*>，并将i,j指向下一个与token相同的词
        if token != seq1[i] or token != seq2[j]:
            retVal.append('<*>')
            i = seq1.index(token, i)
            j = seq2.index(token, j)
        retVal.append(token)
        i = i + 1
        j = j + 1
    if i < len(seq1) or j < len(seq2):
        retVal.append("<*>")
    return retVal
