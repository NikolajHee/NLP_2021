import nltk
from nltk.corpus import stopwords
from nltk.tag.brill import Pos
import random
import string

#Naive bayes with new bag of words representation
#Open review documents
file = open('auto_1stjerner.txt')
textdocument1 = file.read().splitlines()
stjerner1=[]
for a in range(len(textdocument1[2::4])):
    stjerner1.append(list(nltk.tokenize.word_tokenize(textdocument1[2::4][a])))
file.close()

file = open('auto_2stjerner.txt')
textdocument2 = file.read().splitlines()
stjerner2=[]
for a in range(len(textdocument2[2::4])):
    stjerner2.append(list(nltk.tokenize.word_tokenize(textdocument2[2::4][a])))
file.close()

file = open('auto_4stjerner.txt')
textdocument3 = file.read().splitlines()
stjerner4=[]
for a in range(len(textdocument3[2::4])):
    stjerner4.append(list(nltk.tokenize.word_tokenize(textdocument3[2::4][a])))
file.close()

file = open('auto_5stjerner.txt')
textdocument4 = file.read().splitlines()
stjerner5=[]
for a in range(len(textdocument4[2::4])):
    stjerner5.append(list(nltk.tokenize.word_tokenize(textdocument4[2::4][a])))
file.close()

pos_reviews=stjerner4+stjerner5
neg_reviews=stjerner1+stjerner2

stop_words = set(stopwords.words("english"))
stop_words.update(["i",".",",","'","?","’","(",")","-","$","%",":","...","!","1","2","3","4","5","one","two","three","four","five"])
def bag_of_words(words):
	words_clean = []

	for word in words:
		word = word.lower()
		if word not in stop_words and word not in string.punctuation:
			words_clean.append(word)
	
	words_dictionary = dict([word, True] for word in words_clean)
	
	return words_dictionary
#Feature sets
pos_reviews_set = []
for words in pos_reviews:
	pos_reviews_set.append((bag_of_words(words), 'pos'))

neg_reviews_set = []
for words in neg_reviews:
	neg_reviews_set.append((bag_of_words(words), 'neg'))

#Randomize sets
random.shuffle(pos_reviews_set)
random.shuffle(neg_reviews_set)

test_set = pos_reviews_set[:500] + neg_reviews_set[:500]
train_set = pos_reviews_set[500:] + neg_reviews_set[500:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.accuracy(classifier, test_set)
print(accuracy)

custom_review = "im so happy for my delivery"
custom_review_tokens = nltk.word_tokenize(custom_review)
custom_review_set = bag_of_words(custom_review_tokens)
print (classifier.classify(custom_review_set))


prob_result = classifier.prob_classify(custom_review_set)
print (prob_result.prob("neg")) # Output: 0.776128854994
print (prob_result.prob("pos")) # Output: 0.223871145006