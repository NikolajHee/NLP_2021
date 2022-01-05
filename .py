#%%
#imports
import random
import nltk
from nltk.tag.brill import Pos
import numpy as np



#dokumenter

en_stjerner = open("1stjerner.txt")
to_stjerner = open("2stjerner.txt")
fire_stjerner = open("4stjerner.txt")
fem_stjerner = open("5stjerner.txt")

lines_1 = en_stjerner.read().splitlines()[2::4]
lines_2 = to_stjerner.read().splitlines()[2::4]
lines_4 = fire_stjerner.read().splitlines()[2::4]
lines_5 = fem_stjerner.read().splitlines()[2::4]

documents_1 = [(list(nltk.tokenize.word_tokenize(word)), 'neg') for word in lines_1]
documents_2 = [(list(nltk.tokenize.word_tokenize(word)), 'neg') for word in lines_2]
documents_4 = [(list(nltk.tokenize.word_tokenize(word)), 'pos') for word in lines_4]
documents_5 = [(list(nltk.tokenize.word_tokenize(word)), 'pos') for word in lines_5]

en_stjerner.close()
to_stjerner.close()
fire_stjerner.close()
fem_stjerner.close()

documents = documents_1+documents_2+documents_4+documents_5

pos = 0
neg = 0

random.seed(4)
random.shuffle(documents)

for i in range(int(len(documents)/2)):
    if documents[i][1] == "pos":
        pos+=1
    else:
        neg+=1

words = []

for i in range(len(documents)):
    words = words + documents[i][0]




# Define the feature extractor

all_words = nltk.FreqDist(w.lower() for w in words)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features



# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[89:], featuresets[:89]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))


# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(10)


#tre_stjerner = open("5stjerner.txt")
#lines3 = tre_stjerner.read().splitlines()

#for line in lines3:
#    test = line
#    testt = {word: (word in nltk.word_tokenize(test.lower())) for word in all_words}
#    print(test," : ", classifier.classify(testt))



test = "fast fast fast fast"


testt = {word: (word in nltk.word_tokenize(test.lower())) for word in all_words}


print(test," : ", classifier.classify(testt))




# %%
