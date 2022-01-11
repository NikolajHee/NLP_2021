#%%
import numpy as np
import pandas as pd


df = pd.read_csv("Auto/data_set.csv")
df.head()

print(df.shape)

X = df['Data']
y = df['Catagory']

#test - training split

#%%

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)



#%%
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

stopWords = set(stopwords.words("english"))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

def text_cleaning(a):
    remove_punctuation = [char for char in a if char not in string.punctuation]
    remove_punctuation=''.join(remove_punctuation)
    return [word for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]


print(df.iloc[:,1].apply(text_cleaning))

#%%
bow_transformer = CountVectorizer(analyzer=text_cleaning).fit(list(X_train))

bow_transformer.vocabulary_
# %%

title_bow = bow_transformer.transform(list(X_train))
print(title_bow)
# %%

from sklearn.feature_extraction.text import TfidfTransformer


tfidf_transformer = TfidfTransformer().fit(title_bow)
print(tfidf_transformer)

title_tfidf = tfidf_transformer.transform(title_bow)
print(title_tfidf)


# %%

from sklearn.naive_bayes import MultinomialNB


model = MultinomialNB().fit(title_tfidf,list(y_train))


predictions = model.predict(bow_transformer.transform(list(X_test)))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


print(model.predict(bow_transformer.transform(["this shoe was so bad. didnt come in time, size was wrong"])))


print(model.predict(bow_transformer.transform(["my dress fitted perfectly, it was beautiful"])))

# %%

# %%
