
import csv
import panda as pd


#f = open('Auto/data_set.csv','w')

#writer = csv.writer(f)



#%%


df = pd.read_csv("data_set.csv")

df.drop([], axis=0, inplace = True)

df.to_csv("Auto/data_set.csv", index = False)


#%%


#adding space after punctuations and alike
def space_after_symbols(document):
    file = open("document")
    


#%%


#Testing length of data-set

category = 'pos'
word_count = 0

for line in list(df['Data'][list(df['Category']==category)]):
    word_count += len(line.split())

print(word_count)
# %%
