from afinn import Afinn
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import pandas as pd


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

correct=0
for i in range(len(ascore)):
 if ([q[1] for q in ascore][i])==([q[1] for q in all_reviews][i]):
     correct+=1
accuracy=correct/len(ascore)      
#print(accuracy)