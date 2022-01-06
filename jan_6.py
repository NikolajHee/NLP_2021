#%%
#imports
import random
import nltk
from nltk.tag.brill import Pos
import numpy as np
from nltk.corpus import movie_reviews




#dokumenter

documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

random.seed(4)
random.shuffle(documents)



words = []

for i in range(len(documents)):
    words = words + documents[i][0]





# Define the feature extractor

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features



# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[1222:], featuresets[:1222]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))


# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(120)


#tre_stjerner = open("5stjerner.txt")
#lines3 = tre_stjerner.read().splitlines()

#for line in lines3:
#    test = line
#    testt = {word: (word in nltk.word_tokenize(test.lower())) for word in all_words}
#    print(test," : ", classifier.classify(testt))

#%%

test = "Order was delivered in 8 days, package presentation was what you would expect from price point. I was having a really hard time finding shoe in the size that I needed. I would absolutely order from FF again under the same circumstances. Thanks for the super quick shipment."

testt = {word: (word in nltk.word_tokenize(test.lower())) for word in all_words}

print(test," : ", classifier.classify(testt))

# %%
