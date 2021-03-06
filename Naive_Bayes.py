

#NAIVE BAYES MODEL
#-------------------------------------------------------------------------------------
#Nikolaj og Gustav




#imports
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve,auc
import matplotlib.patches as patches
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import (precision_score,recall_score,f1_score)
import pandas as pd


#Loading data into variables:
path = "data_set.csv" 

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
#https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
vectorizer = TfidfVectorizer(max_features=2500, min_df=5, max_df=0.8, stop_words=stopwords.words('english'), ngram_range = (1,2)) 
processed_features = vectorizer.fit_transform(reviews).toarray()

#Defining model (Naive Bayes)
text_classifier = MultinomialNB()

#defining k-folds
k = 5
random_state=None
kf = StratifiedKFold(n_splits = k, random_state=random_state)

#variables
acc_score = []
score = np.zeros(len(processed_features))


#subplots
f, axes = plt.subplots(2, figsize=(8, 16))


#https://www.kaggle.com/kanncaa1/roc-curve-with-k-fold-cv
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

#numbers
pre_neg = []
pre_pos = []
rec_neg =[]
rec_pos =[]
fscore_neg =[]
fscore_pos = []

#k-folds cross-validation loop
#https://www.askpython.com/python/examples/k-fold-cross-validation
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

    #https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    #https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    #PR-data
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, preds, pos_label = 'pos')
    lr_f1, lr_auc = f1_score(y_test, pred_values, pos_label = 'pos'), auc(lr_recall, lr_precision)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    axes[1].step(lr_recall, lr_precision, label='FOLD %d AUC=%.2f' % (j, auc(lr_recall, lr_precision)))
    y_real.append(y_test)
    y_proba.append(preds)
    j += 1

    numbers = classification_report(y_test,pred_values)
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
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(true_labels,prediction), display_labels=["neg","pos"])
disp.plot()
plt.title('Confusion matrix: Naive Bayes')
plt.show()


print("Average model accuracy: ", avg_acc_score, "\n")


#calculating recall, precision and f1-score, along with their confidence intervals with scikit-learn
precision_pos = precision_score(true_labels,prediction,pos_label="pos")
recall_pos = recall_score(true_labels,prediction,pos_label="pos")
f1_pos = f1_score(true_labels,prediction,pos_label="pos")

precision_neg = precision_score(true_labels,prediction,pos_label="neg")
recall_neg = recall_score(true_labels,prediction,pos_label="neg")
f1_neg = f1_score(true_labels,prediction,pos_label="neg")

print("precision_pos: ", precision_pos)
print("[",precision_pos-1.96*np.sqrt(precision_pos*(1-precision_pos)/1605)," ; ",precision_pos+1.96*np.sqrt(precision_pos*(1-precision_pos)/1605),"]")

print("precision_neg: ", precision_neg)
print("[",precision_neg-1.96*np.sqrt(precision_neg*(1-precision_neg)/1605)," ; ",precision_neg+1.96*np.sqrt(precision_neg*(1-precision_neg)/1605),"]")

print("recall_pos: ", recall_pos)
print("[",recall_pos-1.96*np.sqrt(recall_pos*(1-recall_pos)/1605)," ; ",recall_pos+1.96*np.sqrt(recall_pos*(1-recall_pos)/1605),"]")

print("recall_neg: ", recall_neg)
print("[",recall_neg-1.96*np.sqrt(recall_neg*(1-recall_neg)/1605)," ; ",recall_neg+1.96*np.sqrt(recall_neg*(1-recall_neg)/1605),"]")


print("f1_pos: ", f1_pos)
print("[",f1_pos-1.96*np.sqrt(f1_pos*(1-f1_pos)/1605)," ; ",f1_pos+1.96*np.sqrt(f1_pos*(1-f1_pos)/1605),"]")


print("f1_neg: ", f1_neg, "\n")
print("[",f1_neg-1.96*np.sqrt(f1_neg*(1-f1_neg)/1605)," ; ",f1_neg+1.96*np.sqrt(f1_neg*(1-f1_neg)/1605),"]")



#Confidence interval
lower_bound = avg_acc_score - 1.96 * np.sqrt((avg_acc_score*(1-avg_acc_score)/1605))
upper_bound = avg_acc_score + 1.96 * np.sqrt((avg_acc_score*(1-avg_acc_score)/1605))
print("Confidence Interval: [",lower_bound, ";", upper_bound, '] \n')


#examples
line1 = "my dress fitted perfectly, it was beautiful"
line2 = "THIS IS A STUPID WEBSITE. MY SHIRT LOOKS SO BAD ITS AWFUL."

print(line1,":", text_classifier.predict(vectorizer.transform([line1]))[0])

print(line2,":", text_classifier.predict(vectorizer.transform([line2]))[0])



print("\n")

#Get most relevant 
neg_class_prob_sorted = text_classifier.feature_log_prob_[0, :].argsort()[::-1]
pos_class_prob_sorted = text_classifier.feature_log_prob_[1, :].argsort()[::-1]


print("15 most relevant neg. words:\n", np.take(vectorizer.get_feature_names_out(), neg_class_prob_sorted[:15]), "\n")
print("15 most relevant pos. words:\n", np.take(vectorizer.get_feature_names_out(), pos_class_prob_sorted[:15]), "\n")
#https://stackoverflow.com/questions/50526898/how-to-get-feature-importance-in-naive-bayes

# %%
