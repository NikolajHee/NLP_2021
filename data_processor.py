
import csv
import panda as pd


#f = open('Auto/data_set.csv','w')

#writer = csv.writer(f)



#%%


df = pd.read_csv("Auto/data_set.csv")

df.drop([738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755], axis=0, inplace = True)

df.to_csv("Auto/newest_data_set.csv", index = False)


#%%


#adding space after punctuations and alike
def space_after_symbols(document):
    file = open("document")
    
