

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tag.brill import Pos
import string
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer


#Loading reviews
file = open('Auto/auto_4stjerner.txt'); 
lines4 = file.read().splitlines(); 
file.close()
stjerner4 = lines4[2::4]

file = open('Auto/auto_5stjerner.txt'); 
lines5 = file.read().splitlines(); 
file.close()
stjerner5 = lines5[2::4]

file = open('Auto/auto_1stjerner.txt'); 
lines1 = file.read().splitlines(); 
file.close()
stjerner1 = lines1[2::4]

file = open('Auto/auto_2stjerner.txt'); 
lines2 = file.read().splitlines(); 
file.close()
stjerner2 = lines2[2::4]

pos_reviews=stjerner4+stjerner5
neg_reviews=stjerner1+stjerner2
all_reviews=pos_reviews+neg_reviews
# Find labels
pos_reviews_labels = []
for docs in pos_reviews:
	pos_reviews_labels.append('pos')

neg_reviews_labels = []
for docs in neg_reviews:
	neg_reviews_labels.append('neg')

labels=pos_reviews_labels+neg_reviews_labels

#Vectorise and make model
vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stopwords.words('english')) 
processed_features = vectorizer.fit_transform(all_reviews).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.3, random_state=1)
from sklearn.naive_bayes import CategoricalNB
text_classifier = CategoricalNB()
text_classifier.fit(X_train, y_train)
predictions=text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))



