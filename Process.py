# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:29:05 2020

@author: 118857
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn import datasets
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
def plot_confusion_matrix(CM, target_names,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(CM, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def CM_normalized(CM):
    CM_prob = CM.astype('float')/CM.sum(axis=1)[:, np.newaxis]
    return CM_prob
random_state = 123457
clf = LogisticRegression(class_weight='balanced',
                         fit_intercept=True, 
                         penalty='l1',
                         solver='liblinear',
                         tol=0.0001
                        )
filename = 'iCHEF0410WOE.csv'
df = pd.read_csv(filename)
X = df.ix[:,1:-1].values
y = df.ix[:,-1].values
target_names = ['Bad', 'Good']
np.random.seed(random_state)
idx = np.random.permutation(len(X))
X = X[idx]
y = y[idx]
X_test=X
X_train=X
y_test=y
y_train=y
fontsize = 16
plt.rcParams['axes.labelsize']  = fontsize
plt.rcParams['axes.titlesize']  = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
param_grid = {'C': np.logspace(-3, 1, 20)}
scoring = 'accuracy'
#%%
cv = 5
GridSearch_CV = GridSearchCV(estimator=clf,
                             param_grid=param_grid,
                             scoring=scoring,
                             cv=cv,
                             verbose=1,
                             n_jobs=-1)
GridSearch_CV.get_params().keys() 
GridSearch_CV.fit(X_train, y_train)
clf_best = GridSearch_CV.best_estimator_
print(clf_best)
print(GridSearch_CV.best_params_)
clf_best.fit(X_train, y_train)
print(clf_best.coef_,clf_best.intercept_)
#%%
coef_intercept = clf_best.intercept_
clf_coef = clf_best.coef_
Non_zero_parameters = (clf_coef != 0).mean()
clf_best.score(X_train, y_train)
clf_best.score(X_test, y_test)
y_test_pred = clf_best.predict(X_test)
confusion_matrix(y_test, y_test_pred)
CM = confusion_matrix(y_test, y_test_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(CM)


CM_prob = CM_normalized(CM)
print('Normalized confusion matrix')
print(CM_prob)

plt.figure(figsize=(22, 20))
plt.subplot(121)
plot_confusion_matrix(CM,target_names)
plt.subplot(122)
plot_confusion_matrix(CM_prob,target_names,title='Normalized confusion matrix')
plt.subplots_adjust(left=0.08,right=0.95,bottom=0.08,top=0.95,wspace=0.18,hspace=0.05)
plt.show()
#%%
feature = list(df.columns[1:-1])
KF = KFold(n=len(y_train), n_folds=cv,random_state=random_state)
CV_coef = []
CV_coef_nm = []
for k, (train, validation) in enumerate(KF):
    clf_best.fit(X_train[train], y_train[train])
    CV_coef.append(clf_best.coef_*np.std(X[train], 0))
CV_ALLRank=[]
for i in range(len(CV_coef[0][0])):
    CV_Rank=[]
    for ii in range(len(CV_coef)):        
        CV_Rank.append(CV_coef[ii][0][i])
    CV_Rank.sort()
        #print CV_coef[ii][0][i] 
    #CV_ALLRank.append((CV_Rank[4]-CV_Rank[0])/CV_Rank[2])
    CV_ALLRank.append((CV_Rank[4]-CV_Rank[0]))
plt.figure(224)
plt.ylim(0, 1.4)
for k in range(len(CV_coef)):
    plt.plot(range(len(CV_coef[k][0])),CV_coef[k][0],lw=1, label='CV:%d importances' % (k+1))
plt.bar(list(map(lambda x:x-0.1, range(len(CV_ALLRank)))), CV_ALLRank,width=0.2,lw=1, label='Max-Min', color="b")
plt.title('C='+str(GridSearch_CV.best_params_['C']))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('importances')
plt.xlabel('Feature')
plt.savefig('picture\ML_'+str(random_state))
plt.show()
clf_best.fit(X_train, y_train)
importances = clf_best.coef_*np.std(X_train, 0)
importances = importances.T[:,0]
indices = np.argsort(abs(importances))[::-1]
print("Feature ranking:")
for x in range(X_train.shape[1]):
    print("%d %s %f" % (x+1, feature[indices[x]], importances[indices[x]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="b")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
