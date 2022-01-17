#imports
import numpy as np
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve,auc
import matplotlib.patches as patches
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import pandas as pd


#Loading data into variables:
path = "Auto/data_set.csv" 

df = pd.read_csv(path)
reviews = df['Data'] #All_reviews
labels = df['Category'] #Labels

# =============================================================================
#Simple Cross-validation
#from sklearn.model_selection import cross_val_score
#result = cross_val_score(text_classifier, processed_features, y, cv=5)
#print('Cross-validation scores:{}'.format(result))
#print('Average cross-validation score: {:.4f}'.format(result.mean()))
# =============================================================================


#tf-IDF - Vectorizing
vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'), ngram_range = (1,2)) 
processed_features = vectorizer.fit_transform(reviews).toarray()

#Defining model (Naive Bayes)
text_classifier = MultinomialNB()

#defining k-folds
k = 5
random_state=None
kf = StratifiedKFold(n_splits = k, random_state=random_state)


acc_score = []

score = np.zeros(len(processed_features))


f, axes = plt.subplots(2, figsize=(8, 16))

#ROC - graph
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)


#a = []
j = 1

#PR - graph
y_real = []
y_proba = []

#Confusion-matrix
prediction=[]
true_labels=[]

#k-folds cross-validation loop
for train_index, test_index in kf.split(processed_features, labels):
    X_train, X_test = processed_features[train_index], processed_features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    #fitting the data
    text_classifier.fit(X_train, y_train)
    #using the model to predict the test-data
    pred_values = text_classifier.predict(X_test)

    #saving values for confusion matrix
    for values in pred_values:
        prediction.append(values)
    for cat in y_test:
        true_labels.append(cat)

    
    #Saving the true and false predictions
    i = 0
    for t in test_index:
        if pred_values[i] == list(y_test)[i]:
            score[t] = True
        else:
            score[t] = False
        i+=1

    #saving the accuracy
    acc = accuracy_score(pred_values, y_test)
    acc_score.append(acc)

    #probabilities for curves
    probs = text_classifier.predict_proba(X_test)
    preds = probs[:,1]

    #ROC-data
    fpr, tpr, t = roc_curve(list(y_test), preds, pos_label = 'pos')
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)
    axes[0].plot(fpr, tpr, lw=2, alpha=0.3, label = 'ROC FOLD %d (AUC=%0.2f)' % (j,roc_auc))

    #PR-data
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, preds, pos_label = 'pos')
    lr_f1, lr_auc = f1_score(y_test, pred_values, pos_label = 'pos'), auc(lr_recall, lr_precision)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    axes[1].step(lr_recall, lr_precision, label='FOLD %d AUC=%.2f' % (j, auc(lr_recall, lr_precision)))
    y_real.append(y_test)
    y_proba.append(preds)
    print()
    j += 1

    test = classification_report(y_test,pred_values)
    #confusion_matrix(y_test,pred_values)


#PR-curve
axes[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

y_real = np.concatenate(y_real)
y_proba = np.concatenate(y_proba)
precision, recall, _ = precision_recall_curve(y_real, y_proba, pos_label = 'pos')
lab = 'OVERALL AUC=%.4f' % (auc(recall, precision))
axes[1].step(recall, precision, label=lab, lw=2, color='black')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('PR-curve')
axes[1].legend(loc='lower left', fontsize='small')


#ROC-Curve
axes[0].plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
axes[0].plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC')
axes[0].legend(loc="lower right")
plt.show()

#saving figures
#f.tight_layout()
#f.savefig("ROC&PR",bbox_inches = 'tight', pad_inches=0)

#average accuracy
avg_acc_score = sum(acc_score)/k

list_of_index = np.where(score==False)

#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(true_labels,prediction), display_labels=["neg","pos"])
disp.plot()
plt.title('Confusion matrix: Naive Bayes')
plt.show()


print("Average model accuracy: ", avg_acc_score, "\n")

#print(confusion_matrix(y_test,pred_values))
#print(classification_report(y_test,pred_values))#

precision_pos = (0.99+0.95+0.96+0.82+0.85)/5
recall_pos = (0.71+0.79+0.96+0.96+0.97)/5
f1_pos = (0.82+0.86+0.96+0.88+0.90)/5

precision_neg = (0.77+0.81+0.96+0.95+0.96)/5
recall_neg = (0.99+0.96+0.96+0.78+0.82)/5
f1_neg = (0.87+0.88+0.96+0.85+0.88)/5

print("precision_pos: ", precision_pos)
print("precision_neg: ", precision_neg)
print("recall_pos: ", recall_pos)
print("recall_neg: ", recall_neg)
print("f1_pos: ", f1_pos)
print("f1_neg: ", f1_neg, "\n")

#Confidence interval
lower_bound = avg_acc_score - 1.96 * np.sqrt((avg_acc_score*(1-avg_acc_score)/1605))
upper_bound = avg_acc_score + 1.96 * np.sqrt((avg_acc_score*(1-avg_acc_score)/1605))
print("Confidence Interval: [",lower_bound, ";", upper_bound, '] \n')


#examples
line1 = "my dress fitted perfectly, it was beautiful"
line2 = "THIS IS A STUPID WEBSITE. MY SHIRT LOOKS SO BAD ITS AWFUL."

print(line1,":", text_classifier.predict(vectorizer.transform([line1]))[0])

print(line2,":", text_classifier.predict(vectorizer.transform([line2]))[0])



#print(confusion_matrix(y_test,pred_values))
#a.append(classification_report(y_test,pred_values))
#print(text_classifier.predict(vectorizer.transform([df['Data'][7]])))
# %%
