#%%
#imports
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer


#Loading:
import pandas as pd

path = "Auto/data_set.csv" 

df = pd.read_csv(path)
X = df['Data'] #All_reviews
y = df['Category'] #Labels


from sklearn.model_selection import KFold,StratifiedKFold
k = 5
random_state=None
kf = StratifiedKFold(n_splits = k, random_state=None)


#Vectorize
vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'), ngram_range = (1,2)) 
processed_features = vectorizer.fit_transform(X).toarray()



from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


text_classifier = MultinomialNB()

#result = cross_val_score(text_classifier, processed_features, y, cv=5)

acc_score = []

score = np.zeros(len(processed_features))



for train_index, test_index in kf.split(processed_features, y):
    X_train, X_test = processed_features[train_index], processed_features[test_index]
    y_train, y_test = y[train_index], y[test_index]

    text_classifier.fit(X_train, y_train)
    pred_values = text_classifier.predict(X_test)
    #print(test_index)
    i = 0
    for t in test_index:
        if pred_values[i] == list(y_test)[i]:
            score[t] = True
        else:
            score[t] = False
        i+=1

    acc = accuracy_score(pred_values, y_test)
    acc_score.append(acc)

    predictions = text_classifier.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    #print(text_classifier.predict(vectorizer.transform([df['Data'][7]])))

   
avg_acc_score = sum(acc_score)/k

list_of_index = np.where(score==False)




print(avg_acc_score)

#

#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))#
##print(accuracy_score(y_test, predictions))



Line1 = "my dress fitted perfectly, it was beautiful"

Line2 = "FUCK THIS SHITTY WEBSITE. MY SHIRT LOOKS LIKE CRAP ITS AWFUL."

print(Line1,":", text_classifier.predict(vectorizer.transform([Line1]))[0])

print(Line2,":", text_classifier.predict(vectorizer.transform([Line2]))[0])


#print(text_classifier.predict(vectorizer.transform(["my dress was too msall, and the box was horrible"])))

#print(text_classifier.predict(vectorizer.transform([df['Data'][7]])))




# %%
