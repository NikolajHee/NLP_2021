#imports
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tag.brill import Pos
import string
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


df = pd.read_csv("Auto/data_set.csv")

all_reviews = df['Data']
labels = df['Category']


#Vectorise and make model
vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'), ngram_range = (1,2)) 
processed_features = vectorizer.fit_transform(all_reviews).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.3, random_state=1, stratify=labels)

from sklearn.naive_bayes import MultinomialNB

text_classifier = MultinomialNB()
text_classifier.fit(X_train, y_train)
predictions=text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


#%%




print(text_classifier.predict(vectorizer.transform(["my dress fitted perfectly, it was beautiful"])))

print(text_classifier.predict(vectorizer.transform(["my dress was too mall, and the box was horrible"])))



# %%
