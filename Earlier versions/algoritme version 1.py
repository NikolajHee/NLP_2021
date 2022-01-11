
import nltk
from nltk.corpus import stopwords
from nltk.tag.brill import Pos

file = open('1stjerner.txt')
textdocument = file.read()
words = nltk.tokenize.word_tokenize(textdocument)
file.close()



#AFINN RATING (MAKING TRAINING DATASET)
from afinn import Afinn
import numpy as np

afinn = Afinn()

file = open('1stjerner.txt')
lines = file.read().splitlines()

title = lines[0::4]
text = lines[2::4]
file.close()

scores = []


for i in range(len(title)):
    score = afinn.score(title[i]+' '+text[i])
    if score > 0:
        scores.append('pos')
    else:
        scores.append('neg')

tokeText = []

for a in range(len(text)):
    tokeText.append(list(nltk.tokenize.word_tokenize(text[a])))


dataSet = list(zip(tokeText,scores))



#REMOVING NON-IMPORTANT WORDS
stop_words = set(stopwords.words("english"))
stop_words.update(["i",".",",","'","?","’","(",")","-","$","%",":","..."])


filtered_words = []

for w in words:
    if w not in stop_words:
        filtered_words.append(w)



freq_all_words = nltk.FreqDist(w.lower() for w in filtered_words)
bag_of_words = list(all_words)[:2000]


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