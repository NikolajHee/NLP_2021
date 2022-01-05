#%%
#imports
import random
import nltk
from nltk.tag.brill import Pos
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import word_tokenize



#dokumenter

en_stjerner = open("1stjerner.txt")
to_stjerner = open("2stjerner.txt")
fire_stjerner = open("4stjerner.txt")
fem_stjerner = open("5stjerner.txt")

lines_1 = en_stjerner.read().splitlines()[2::4]
lines_2 = to_stjerner.read().splitlines()[2::4]
lines_4 = fire_stjerner.read().splitlines()[2::4]
lines_5 = fem_stjerner.read().splitlines()[2::4]

#tokenize

documents_1 = [(list(nltk.tokenize.word_tokenize(word)), 'neg') for word in lines_1]
documents_2 = [(list(nltk.tokenize.word_tokenize(word)), 'neg') for word in lines_2]
documents_4 = [(list(nltk.tokenize.word_tokenize(word)), 'pos') for word in lines_4]
documents_5 = [(list(nltk.tokenize.word_tokenize(word)), 'pos') for word in lines_5]

documents = documents_1+documents_2+documents_4+documents_5

stop_words = set(stopwords.words('english'))



words2 = []

for i in range(len(documents)):
    words2 = words2 + documents[i][0]
print(len(words2))

for tuples in documents:
    t = 0
    for w in tuples[0]:
        if w in stop_words:
            tuples[0].pop(t)
            #print(w)
        else:
            t+=1


#nltk.download('punkt')
ps = nltk.stem.PorterStemmer()



#tokens = [ps.stem(_) for _ in word_tokenize(text[index])]

#print(tokens)


en_stjerner.close()
to_stjerner.close()
fire_stjerner.close()
fem_stjerner.close()



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
print(len(words))




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
