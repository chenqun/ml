__author__ = 'chenqyu'
import numpy as np
import pandas as pd
import random as rd
import time

def entropy(x, y):
    bin_count = np.bincount(x)
    xkeys, xvalue = np.nonzero(bin_count)[0], bin_count[bin_count > 0]
    # xkeys, xvalue = np.unique(x,return_counts = True)
    entpy = 0
    nsize = len(xkeys)
    for i in range(nsize):
        yt = y[x == xkeys[i]]
        y1 = yt.sum()
        tlen = xvalue[i]
        ny1 = tlen - y1
        if y1 == 0 or ny1 == 0:
            ent = 0
        else:
            ent = -1 * y1 / tlen * np.log2(y1 / tlen)- ny1 / tlen * np.log2(ny1 / tlen)
        entpy += ent * tlen / len(x)
    return entpy

def LeafCal(y):
    y = np.array(y)
    return y.sum()/len(y)
    # xkeys, xvalue = np.nonzero(bin_count)[0], bin_count[bin_count > 0]
    # key,value = np.unique(y,return_counts = True)
    # pos = np.where(value == np.max(value))[0]
    # return key[pos].tolist()[0]

def TwoSplit( x, y):
    # xquan = np.percentile(x,range(0,100))
    # xkey = np.unique(xquan)
    xkey = np.unique(x)
  #  nlen = len(xkey)
    minentr = 2
    nlen = x.shape
    for i in xkey:
        x1 = np.zeros(nlen, dtype=np.int)
        x1[x < i] = 1
        # x1 = [int(j < i) for j in x]
        tmp = entropy(x1, y)
        if tmp < minentr:
            keep = i
            minentr = tmp
            xkeep = x1
    return [keep, minentr, xkeep]

def ValSelection(xlist,y):
    lst_len = xlist.shape[1]
    minimum = 2
    for i in range(lst_len):
        x = xlist[:,i]
        split, ret, xk = TwoSplit(x, y)
        if ret < minimum:
            pos = i
            min_p = split
            minimum = ret
            xkeep = xk
    return [min_p,minimum,pos,xkeep]

def Tree(dlist, nFet = 40):
    nsize = dlist.shape[0]
    ncol = dlist.shape[1]
    dlistp = dlist.sample(nsize, replace=True)
    columns = dlistp.columns.values
    y = dlistp.values[:, 0]
    if len(np.unique(y)) == 1:
        return y[0]
    col = rd.sample(range(1,ncol),nFet)
 #   xlist = dlistp.values[:,col]
    colt = columns[col]
    min_p, minimum, pos, xkeep = ValSelection(dlistp.values[:,col], y)
    valset = np.unique(xkeep)
    if len(valset) == 1:
        return LeafCal(y)
    tree = {colt[pos]: {}}
    tree[colt[pos]]['ALL'] = LeafCal(y)
    tree[colt[pos]]['split'] = min_p
    subset1 = dlist[dlist[colt[pos]] < min_p]
    subset2 = dlist[dlist[colt[pos]] >= min_p]
    tree[colt[pos]][-1] = Tree(subset1)
    tree[colt[pos]][1] = Tree(subset2)
    return tree

def RandomForest(dlist,ntree = 200, nFet = 40):
    Ranf = []
    for i in range(ntree):
        print(i)
        a = Tree(dlist, nFet)
        Ranf.append(a)
        print(a)
    return Ranf

def Prediction(x, model):
    pos = list(model.keys())[0]
    if x[pos] < model[pos]['split'] and isinstance(model[pos][-1], dict):
        return Prediction(x, model[pos][-1])
    elif x[pos] < model[pos]['split'] and -1 in model[pos]:
        return model[pos][-1]
    elif x[pos] >= model[pos]['split'] and isinstance(model[pos][1], dict):
        return Prediction(x, model[pos][1])
    elif x[pos] >= model[pos]['split'] and 1 in model[pos]:
        return model[pos][1]
    else:
        return model[pos]['ALL']


def RFPrediction(x, model):
    ntree = len(model)
    pred = []
    for i in range(ntree):
        pred.append(Prediction(x,model[i]))
    return LeafCal(pred)

def RFPredictData(xlist, model):
    nrow = xlist.shape[0]
    ypred = []
    for i in range(nrow):
        ypred.append(RFPrediction(xlist.iloc[i],model))
    return ypred

def CalAccuracy(ytrue,ypred):
    return (np.array(ytrue == ypred).sum()/len(ytrue))

train = pd.read_csv('c:/train.csv')
test = pd.read_csv('c:/test.csv')
pre = time.time()
#traindata = train.values
#TwoSplit(traindata[:,1],traindata[:,0])

#print(Tree(train, nFet = 10))
#print(RandomForest(train))
rfmodel = RandomForest(train)
#y = test.iloc[:,0]
#xlist = test.drop(test.columns[[0]],1)
#train.drop()
pre1 = time.time()
ypred = RFPredictData(test,rfmodel)
output = pd.DataFrame(ypred, columns = ["PredictedProbability"])
output.index = np.arange(1,len(output)+1)
output.to_csv('c:\output.csv', index = True, index_label= "MoleculeId")
print(pre1 - pre)
print(ypred)
#accuracy = CalAccuracy(y,ypred)
#print(accuracy)
post = time.time()
print(post-pre)