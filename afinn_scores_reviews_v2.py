

#AFINN MODEL
#--------------------------------------------------------------------
#Nikolaj og Gustav




#imports
from afinn import Afinn
import numpy as np
import nltk
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd
import collections
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#https://github.com/fnielsen/afinn
#Define afinn
afinn=Afinn()

#Load data
path = 'data_set.csv'
df = pd.read_csv(path)
all_reviews = list(zip(df['Data'],df['Category']))

#Get all scores
afinn_scores=np.zeros(1605)
for i in range(len(all_reviews)):
    a=afinn.score(all_reviews[i][0])
    afinn_scores[i]=a


#Function classifiying a review(text string) as positive or negative
def afinnclassifier(review):
    a=afinn.score(review)
    if (a<0)or(a==0):
     cat='neg'
    elif a>0:
     cat='pos'
    return cat


#Creating set of refernce values and test values + labels and classified labels
referencesets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
labels=[]
tests=[]
for i, (feats, label) in enumerate(all_reviews):
    referencesets[label].add(i)
    observed = afinnclassifier(feats)
    testsets[observed].add(i)
    labels.append(label)
    tests.append(observed)

#Calculate accuracy
correct=0
correct_index=np.zeros(len(labels))
for i in range(len(labels)):
 if labels[i]==tests[i]:
     correct+=1
     correct_index[i]=(1)
 else:
     correct_index[i]=(0)
accuracy=correct/len(labels) 

#Print and calculate different metrics
print ('Accuracy:', accuracy)
print ('pos precision:', nltk.precision(referencesets['pos'], testsets['pos']))
print ('pos recall:', nltk.recall(referencesets['pos'], testsets['pos']))
print ('pos F-measure:', nltk.f_measure(referencesets['pos'], testsets['pos']))
print ('neg precision:', nltk.precision(referencesets['neg'], testsets['neg']))
print ('neg recall:', nltk.recall(referencesets['neg'], testsets['neg']))
print ('neg F-measure:', nltk.f_measure(referencesets['neg'], testsets['neg']))
#print(nltk.ConfusionMatrix(labels,tests))

#Confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(labels,tests), display_labels=["neg","pos"])
disp.plot()
plt.title('Confusion matrix: AFINN')
plt.show()

#Afinn score plot
afinnscorespos=[]
for i in range(13):
    afinnscorespos.append(np.count_nonzero(afinn_scores==i))
t=-12
afinnscoresneg=[]
for i in range(12):
    afinnscoresneg.append(np.count_nonzero(afinn_scores==t))
    t=t+1
    
#histogram showing data distribution
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
etiket = ["-12","-11","-10","-9","-8","-7","-6","-5","-4","-3","-2","-1",'0', '1', '2', '3', '4',"5","6","7","8","9","10","11","12"]
ax.bar(etiket,afinnscoresneg+afinnscorespos)
plt.xlabel("Score")
plt.ylabel("Count")
plt.show()