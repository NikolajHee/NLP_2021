#imports
from sklearn.naive_bayes import BernoulliNB
import nltk
from nltk.corpus import stopwords
from nltk.tag.brill import Pos
import random
import string
import numpy as np
import datetime

#Naive bayes with new bag of words representation
#Open review documents and tokenize words
file = open('Auto/auto_1stjerner.txt')
textdocument1 = file.read().splitlines()
stjerner1=[]
for a in range(len(textdocument1[2::4])):
    stjerner1.append(list(nltk.tokenize.word_tokenize(textdocument1[2::4][a])))
file.close()

file = open('Auto/auto_2stjerner.txt')
textdocument2 = file.read().splitlines()
stjerner2=[]
for a in range(len(textdocument2[2::4])):
    stjerner2.append(list(nltk.tokenize.word_tokenize(textdocument2[2::4][a])))
file.close()

file = open('Auto/auto_4stjerner.txt')
textdocument3 = file.read().splitlines()
stjerner4=[]
for a in range(len(textdocument3[2::4])):
    stjerner4.append(list(nltk.tokenize.word_tokenize(textdocument3[2::4][a])))
file.close()

file = open('Auto/auto_5stjerner.txt')
textdocument4 = file.read().splitlines()
stjerner5=[]
for a in range(len(textdocument4[2::4])):
    stjerner5.append(list(nltk.tokenize.word_tokenize(textdocument4[2::4][a])))
file.close()
#Define positve and negative reviews
pos_reviews=stjerner4+stjerner5
neg_reviews=stjerner1+stjerner2

#Function creating bag of words, removing non-important words and duplicates - string.punctuation og word update overlapper
stop_words = set(stopwords.words("english"))
stop_words.update(["i",".",",","'","?","’","(",")","-","$","%",":","...","!","1","2","3","4","5","one","two","three","four","five",])
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
stop_words__bigrams = set(stop_words) - set(important_words)


def clean_words(words, stopwords_english):
	words_clean = []
	for word in words:
		word = word.lower()
		if word not in stopwords_english and word not in string.punctuation:
			words_clean.append(word)	
	return words_clean

def bag_of_words(words):	
	words_dictionary = dict([word, True] for word in words)	
	return words_dictionary

def bag_of_ngrams(words, n):
	words_ng = []
	for item in iter(nltk.ngrams(words, n)):
		words_ng.append(item)
	words_dictionary = dict([word, True] for word in words_ng)	
	return words_dictionary

def bag_of_all_words(words, n=2):
	words_clean = clean_words(words, stop_words)
	words_clean_for_bigrams = clean_words(words, stop_words__bigrams)

	unigram_features = bag_of_ngrams(words_clean,1)
	bigram_features = bag_of_ngrams(words_clean_for_bigrams,2)

	all_features = unigram_features.copy()
	all_features.update(bigram_features)

	return all_features

#Feature sets
pos_reviews_set = []
for words in pos_reviews:
	pos_reviews_set.append((bag_of_all_words(words), 'pos'))

neg_reviews_set = []
for words in neg_reviews:
	neg_reviews_set.append((bag_of_all_words(words), 'neg'))

#Randomize sets
random.seed(0)
random.shuffle(pos_reviews_set)
random.shuffle(neg_reviews_set)

#Make naive bayes classifier and see accuracy
test_set = pos_reviews_set[600:] + neg_reviews_set[600:]
train_set = pos_reviews_set[:600] + neg_reviews_set[:600]

classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.accuracy(classifier, test_set)
print(accuracy)

#Custom review and its probability of neg and pos
custom_review = "Once I got started with my order everything went smoothly. The only drawback for me is not excepting American Express because that is my main form of payment. Marion"
custom_review_tokens = nltk.word_tokenize(custom_review)
custom_review_set = bag_of_all_words(custom_review_tokens)
print (classifier.classify(custom_review_set))

file = open("output.txt", 'a')
date = datetime.datetime.now()
file.write(str(date.day)+'-'+str(date.month)+'-'+str(date.year)+'--'+str(date.hour)+':'+str(date.minute)+" : Accuracy: "+ str(accuracy) + '\n')

file.close()


prob_result = classifier.prob_classify(custom_review_set)
print (prob_result.prob("neg"))
print (prob_result.prob("pos"))


clf = BernoulliNB()
clf.fit()
