#%%
#imports
import numpy as np
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pylab as plt
from scipy import interp
from sklearn.metrics import roc_curve,auc
import matplotlib.patches as patches
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score




#Loading:
import pandas as pd

path = "Auto/data_set.csv" 

df = pd.read_csv(path)
X = df['Data'] #All_reviews
y = df['Category'] #Labels



k = 5
random_state=None
kf = StratifiedKFold(n_splits = k, random_state=None)


#Vectorize
vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'), ngram_range = (1,2)) 
processed_features = vectorizer.fit_transform(X).toarray()






text_classifier = MultinomialNB()

#result = cross_val_score(text_classifier, processed_features, y, cv=5)

acc_score = []

score = np.zeros(len(processed_features))

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
a = []
j = 1
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

    probs = text_classifier.predict_proba(X_test)
    preds = probs[:,1]


    predictions = text_classifier.predict(X_test)
    #print(confusion_matrix(y_test,predictions))
    #a.append(classification_report(y_test,predictions))
    #print(text_classifier.predict(vectorizer.transform([df['Data'][7]])))
    fpr, tpr, t = roc_curve(list(y_test), preds, pos_label = 'pos')
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label = 'ROC FOLD %d (AUC=%0.2f)' % (j,roc_auc))


    j += 1

plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
#plt.text(0.32,0.7,'More accurate area',fontsize = 12)
#plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
plt.show()

avg_acc_score = sum(acc_score)/k

list_of_index = np.where(score==False)




print(avg_acc_score)

#

#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))#

precision_pos = (0.99+0.95+0.96+0.82+0.85)/5
recall_pos = (0.71+0.79+0.96+0.96+0.97)/5
f1_pos = (0.82+0.86+0.96+0.88+0.90)/5

precision_neg = (0.77+0.81+0.96+0.95+0.96)/5
recall_neg = (0.99+0.96+0.96+0.78+0.82)/5
f1_neg = (0.87+0.88+0.96+0.85+0.88)/5

#print(precision_pos,recall_pos, f1_pos)

#print(precision_neg, recall_neg, f1_neg)

##print(accuracy_score(y_test, predictions))



#Line1 = "my dress fitted perfectly, it was beautiful"
#Line2 = "FUCK THIS SHITTY WEBSITE. MY SHIRT LOOKS LIKE CRAP ITS AWFUL."

#testt = text_classifier.predict(vectorizer.transform([Line1]))[0]
#print(Line2,":", text_classifier.predict(vectorizer.transform([Line2]))[0])


#print(text_classifier.predict(vectorizer.transform(["my dress was too msall, and the box was horrible"])))

#print(text_classifier.predict(vectorizer.transform([df['Data'][7]])))


#Confidence interval
#lower_bound = avg_acc_score - 1.96 * np.sqrt((avg_acc_score*(1-avg_acc_score)/1605))
#upper_bound = avg_acc_score + 1.96 * np.sqrt((avg_acc_score*(1-avg_acc_score)/1605))
#print("[",lower_bound, ";", upper_bound, ']')
# %%
