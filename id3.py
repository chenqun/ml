ｓ__author__ = 'chenqyu'

import numpy as np
import pandas as pd
import time

def entropy(x, y):
    val1 = {}
    val2 = {}
    base = len(y)
    for i in range(base):
        if x.iloc[i] not in val1.keys():
            val1[x.iloc[i]] = {}
            val1[x.iloc[i]][y.iloc[i]] = 1
            val2[x.iloc[i]] = 1
        else:
            if y.iloc[i] not in val1[x.iloc[i]].keys():
                val1[x.iloc[i]][y.iloc[i]] = 1
            else:
                val1[x.iloc[i]][y.iloc[i]] += 1
            val2[x.iloc[i]] += 1
    entpy = 0
    for i in val1.keys():
        probt = np.array([val1[i][j] / val2[i] for j in val1[i].keys()])
        entpyt = (-1 * probt * np.log2(probt)).sum()
        entpy += entpyt * val2[i] / base
    return entpy


def ValSelection(dlist):
    lst_len = dlist.shape[1]-1
    minimum = 2
    for i in range(lst_len):
        y = dlist.iloc[:,lst_len]
        xlist = dlist.iloc[:,i]
        ret = entropy(xlist, y)
        if ret < minimum:
            pos = i
            minimum = ret
    return dlist.columns[pos]

def LeafCal(y):
    outp = {}
    maximum = 0
    for i in y:
        if i not in outp.keys():
            outp[i] = 1
        else:
            outp[i] += 1
    for i in outp.keys():
        if outp[i] > maximum:
            maximum = outp[i]
            ret = i
    return ret

def ID3(dlist):
    if len(dlist.shape) == 1:
        return LeafCal(dlist)
    ncol = dlist.shape[1]
    y = dlist.iloc[:, (ncol-1)]
    if len(y.unique()) == 1:
        return y.iloc[0]
    pos = ValSelection(dlist)
    valset = dlist[pos].unique()
    tree = {pos: {}}
    tree[pos]['ALL'] = LeafCal(y)
    for i in valset:
        subset = dlist[dlist[pos] == i]
        subset = subset.drop(pos,1)
        tree[pos][i] = ID3(subset)
    return tree

def Prediction(x, model):
    pos = list(model.keys())[0]
    if x[pos] in model[pos]:
        sub_model = model[pos][x[pos]]
        if( isinstance(sub_model, dict)):
            return Prediction(x, sub_model)
        else:
            return sub_model
    else:
        return model[pos]['ALL']

def CalAccuracy(ytrue,ypred):
    return (np.array(ytrue == ypred).sum()/len(ypred))

def main():
    pre_time = time.time()
    train = pd.read_csv('c:/train.txt')
    test = pd.read_csv('c:/Ftest.txt')
    ret2 = ID3(train)
    print(ret2)
    ypred = []
    nrow = test.shape[0]
    ncol = test.shape[1]
    for i in range(nrow):
        ypred.append(Prediction(test.iloc[i],ret2))
    accuracy = CalAccuracy(test.iloc[:,(ncol-1)],ypred)
    print(accuracy)
    post_time = time.time()
    print(post_time - pre_time)
main()

###ret2 = ID3(dataSet)
###ret3 = Prediction([1,1], ret2)
###print(ret2,ret3)
