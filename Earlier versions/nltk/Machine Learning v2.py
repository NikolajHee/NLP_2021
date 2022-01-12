
import nltk
from nltk.corpus import stopwords
from nltk.tag.brill import Pos

#fileName = "1stjerner.txt"


file = open("samlet dokument.txt")
textdocument = file.read()
words = nltk.tokenize.word_tokenize(textdocument)
file.close()


#RATING BASED ON STARS

import numpy as np


file = open('1stjerner.txt')
lines1 = file.read().splitlines()

#title = lines[0::4]
text1 = lines1[2::4]
file.close()
stars1 = len(text1)

tokeText1 = []

for i in range(len(text1)):
    tokeText1.append(list(nltk.tokenize.word_tokenize(text1[i])))


stars1 = stars1 * ["neg"]

file = open('2stjerner.txt')
lines2 = file.read().splitlines()

#title2 = lines[0::4]
text2 = lines2[2::4]
file.close()
stars2 = len(text2)

tokeText2 = []

for i in range(len(text2)):
    tokeText2.append(list(nltk.tokenize.word_tokenize(text2[i])))

stars2 = stars2 * ["neg"]

file = open('4stjerner.txt')
lines4 = file.read().splitlines()

#title3 = lines[0::4]
text4 = lines4[2::4]
file.close()
stars4 = len(text4)

tokeText4 = []

for i in range(len(text4)):
    tokeText4.append(list(nltk.tokenize.word_tokenize(text4[i])))

stars4 = stars4 * ["pos"]

file = open('5stjerner.txt')
lines5 = file.read().splitlines()

#title = lines[0::4]
text5 = lines5[2::4]
file.close()
stars5 = len(text5)

tokeText5 = []

for i in range(len(text5)):
    tokeText5.append(list(nltk.tokenize.word_tokenize(text1[5])))

stars5 = stars5 * ["pos"]



tokeText = tokeText1+tokeText2+tokeText4+tokeText5

scores = stars1+stars2+stars4+stars5

dataSet = list(zip(tokeText,scores))



#REMOVING NON-IMPORTANT WORDS
stop_words = set(stopwords.words("english"))
stop_words.update(["&","i",".",",","'","?","’","(",")","-","$","%",":","...",'a',"!","also"])


filtered_words = []

for w in words:
    if w.lower() not in stop_words:
        filtered_words.append(w)



freq_all_words = nltk.FreqDist(w.lower() for w in filtered_words)
bag_of_words = list(freq_all_words)[:2000]


def document_features(document):
    document_words = set(document)
    features = {}
    for word in bag_of_words:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in dataSet]
train_set, test_set = featuresets[25:], featuresets[:25]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))


test = "go to the refused leather"


testt = {word: (word in nltk.word_tokenize(test.lower())) for word in freq_all_words}


print(test," : ", classifier.classify(testt))