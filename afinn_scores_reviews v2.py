from afinn import Afinn
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import nltk
from nltk.metrics.scores import (precision, recall, f_measure)
from nltk.metrics import ConfusionMatrix
import collections

afinn=Afinn()

path = 'current_data_set1.csv'
df = pd.read_csv(path)

all_reviews = list(zip(df['Data'],df['Category']))


#Calculate accuracy
ascore=[]
for i in range(len(all_reviews)):
 a=afinn.score(all_reviews[i][0])
 if (a<0)or (a==0):
     cat='neg'
 elif a>0:
     cat='pos'
 ascore.append((a,cat))


def afinnclassifier(review):
    a=afinn.score(review)
    if (a<0)or(a==0):
     cat='neg'
    elif a>0:
     cat='pos'
    return cat


correct=0
indexwrong=[]
for i in range(len(ascore)):
 if ([q[1] for q in ascore][i])==([q[1] for q in all_reviews][i]):
     correct+=1
 else:
     indexwrong.append(i)
     
accuracy=correct/len(ascore)      
#print(accuracy)

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
labels=[]
tests=[]
for i, (feats, label) in enumerate(all_reviews):
    refsets[label].add(i)
    observed = afinnclassifier(feats)
    testsets[observed].add(i)
    labels.append(label)
    tests.append(observed)

print ('pos precision:', nltk.precision(refsets['pos'], testsets['pos']))
print ('pos recall:', nltk.recall(refsets['pos'], testsets['pos']))
print ('pos F-measure:', nltk.f_measure(refsets['pos'], testsets['pos']))
print ('neg precision:', nltk.precision(refsets['neg'], testsets['neg']))
print ('neg recall:', nltk.recall(refsets['neg'], testsets['neg']))
print ('neg F-measure:', nltk.f_measure(refsets['neg'], testsets['neg']))
print(nltk.ConfusionMatrix(labels,tests)) 