import csv
import pandas as pd
import random

df = pd.read_csv("Auto/data_set.csv")

all_reviews = list(zip(df['Data'],df['Category']))

random.seed(0)
random.shuffle(all_reviews)
random.seed(1)
random.shuffle(all_reviews)
random.seed(2)
random.shuffle(all_reviews)

print("Velkommen. Hedder du Nikolaj eller Gustav: ")
read = input()
if read.lower() == "nikolaj":
    data = all_reviews[:(int(len(all_reviews)/2))]
elif read.lower() == "gustav":
    data = all_reviews[(int(len(all_reviews)/2)):]


print("Sæt dig godt til rette, her kommmer {} spørgsmål.".format(int(384/2)))

scores = 0

for i in range(int(384/2)):
    print("Nummer {}: \n".format(i+1))
    print(data[i][0])
    print("p el. n:")
    clas = input()
    if clas == 'p':
        answer = 'pos'
    elif clas == 'n':
        answer = 'neg'
    if answer == data[i][1]:
        scores += 1
    

print("Tak for spillet. Din score blev: {} ud af {}".format(scores, int(384/2)))