
import nltk
from nltk.corpus import stopwords
from nltk.tag.brill import Pos


from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)







#REMOVING NON-IMPORTANT WORDS
##stop_words = set(stopwords.words("english"))
#stop_words.update(["&","i",".",",","'","?","’","(",")","-","$","%",":","...",'a',"!","also"])


#filtered_words = []

#for w in words:
#    if w.lower() not in stop_words:
#        filtered_words.append(w)



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
train_set, test_set = featuresets[25:], featuresets[:25]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

test = "the movie was outstanding"


testt = {word: (word in nltk.word_tokenize(test.lower())) for word in all_words}


print(test," : ", classifier.classify(testt))