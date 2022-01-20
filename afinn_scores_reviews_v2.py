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

#https://github.com/fnielsen/afinn

afinn=Afinn()

path = 'Auto/data_set.csv'
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
     indexwrong.append(1)
 else:
     indexwrong.append(0)
     
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

print(accuracy)
#Confidence interval
lower_bound = accuracy - 1.96 * np.sqrt((accuracy*(1-accuracy)/1605))
upper_bound = accuracy + 1.96 * np.sqrt((accuracy*(1-accuracy)/1605))
print("[",lower_bound, ";", upper_bound, ']')


x = [i[0] for i in ascore]

plt.hist(x, bins = [-27.5,-22.5,-17.5,-12.5,-7.5,-2.5,2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5,42.5,47.5])



#%%
from sklearn.metrics import roc_curve,auc

fpr, tpr, t = roc_curve([a[1] for a in all_reviews],indexwrong, pos_label = 'pos')

roc_auc = auc(fpr,tpr)

plt.plot(fpr, tpr, lw=2, alpha=0.3, label = 'ROC FOLD %d (AUC=%0.2f)' % (j,roc_auc))

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

# %%
