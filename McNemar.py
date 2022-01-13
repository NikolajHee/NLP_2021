from os import P_PGID
from afinn_scores_reviews_v2 import indexwrong
from INDEX_leg_med_scikitlearn import score
import numpy as np

index = np.arange(1,len(indexwrong)+1)

array = np.array(indexwrong)

#table = np.transpose(np.vstack((index,score,array))).astype(int)

#counting

PP = 0
PF = 0
FF = 0
FP = 0

for i in (index-1):
    if array[i] == 1 and score[i] == 1:
        PP += 1
    if array[i]==1 and score[i]==0:
        PF += 1
    if array[i]==0 and score[i] == 1:
        FP += 1
    if array[i]==0 and score[i] == 0:
        FF += 1

#print(PP,PF,FP,FF)

#%%
from texttable import Texttable
t = Texttable()
t.add_rows([['','AFINN cor', 'AFINN incor'], ['ML cor', PP, FP], ['ML incor',PF,FF ]])
print(t.draw())

#%%------------------------------------------------------
#McNemar

#statistic = ((FP-PF)**2)/(PF+FP)

from statsmodels.stats.contingency_tables import mcnemar

table = [[PP,FP],[PF,FF]]

result = mcnemar(table, exact = False, correction = True)
#correction ved store datasæt

print("Statistic: {}; p-Value {}".format(result.statistic,result.pvalue))

alpha = 0.05

if result.pvalue > alpha:
    print("fail to reject h0") #Der er en signifikant forskel
elif result.pvalue < alpha:
    print("reject h0")


# %%
