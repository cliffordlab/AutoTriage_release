# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:00:02 2020

@author: chait
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics
from joblib import dump, load

# Get histogram of each channel, bins=6
def get_hist(img):
    
    img = cv2.imread(img)
    size1,size2,_ = img.shape
    R = img[:,:,0].reshape(size1*size2)
    G = img[:,:,1].reshape(size1*size2)
    B = img[:,:,2].reshape(size1*size2)
    #bins = [15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255]
    

    R_hist, R_bins = np.histogram(R, bins=6)
    G_hist, G_bins = np.histogram(G, bins=6)
    B_hist, B_bins = np.histogram(B, bins=6)

    R_hist = [a/(size1*size2) for a in R_hist]
    G_hist = [a/(size1*size2) for a in G_hist]
    B_hist = [a/(size1*size2) for a in B_hist]
    
    data = np.concatenate((R_hist,G_hist),axis=0)
    data = np.concatenate((data,B_hist),axis=0)
    
    return data

# LOOCV - prepare data
PATH = "data//kfold"
labels = []
hist = []
for file in os.listdir(PATH):
    if file[0] == 'n':
        labels.append(0)
    else:
        labels.append(1)
    hist.append(get_hist(os.path.join(PATH,file)))
    
# loocv with logistic regression
loocv = LeaveOneOut()
correct = 0
wrong = 0
tp = 0
fp = 0
tn = 0
fn = 0
lr_pred_labels = []
for train_idx, test_idx in loocv.split(hist):
    train_hist = []
    train_labels = []
    for i in train_idx:
        train_hist.append(hist[i])
        train_labels.append(labels[i])
    lr = LogisticRegression(random_state=0, solver='lbfgs').fit(train_hist, train_labels)
    pred = lr.predict([hist[test_idx[0]]])
    lr_pred_labels.append(pred[0])
    if pred[0] == labels[test_idx[0]]:
        correct += 1
        if pred[0] == 1:
            tp += 1
        else:
            tn += 1
    else:
        wrong += 1
        if pred[0] == 1:
            fn += 1
        else:
            fp += 1
accuracy = correct/len(labels)
#print('LR accuracy, tp, fp, tn, fn:', accuracy, tp, fp, tn, fn)

# loocv with knn - 71.43
loocv = LeaveOneOut()
correct = 0
wrong = 0
tp = 0
fp = 0
tn = 0
fn = 0
knn_pred_labels = []
for train_idx, test_idx in loocv.split(hist):
    train_hist = []
    train_labels = []
    for i in train_idx:
        train_hist.append(hist[i])
        train_labels.append(labels[i])
    knn = KNeighborsClassifier(n_neighbors=3).fit(train_hist, train_labels)
    pred = knn.predict([hist[test_idx[0]]])
    knn_pred_labels.append(pred[0])
    if pred[0] == labels[test_idx[0]]:
        correct += 1
        if pred[0] == 1:
            tp += 1
        else:
            tn += 1
    else:
        wrong += 1
        if pred[0] == 1:
            fn += 1
        else:
            fp += 1
accuracy = correct/len(labels)
#print('KNN accuracy, tp, fp, tn, fn:', accuracy, tp, fp, tn, fn)

# loocv with svm - 77
loocv = LeaveOneOut()
correct = 0
wrong = 0
tp = 0
fp = 0
tn = 0
fn = 0
svc_pred_labels = []
for train_idx, test_idx in loocv.split(hist):
    train_hist = []
    train_labels = []
    for i in train_idx:
        train_hist.append(hist[i])
        train_labels.append(labels[i])
    svc = SVC(C=2,gamma='scale',degree=3,probability=True)
    svc.fit(train_hist, train_labels)
    pred = svc.predict([hist[test_idx[0]]])
    svc_pred_labels.append(pred[0])
    if pred[0] == labels[test_idx[0]]:
        correct += 1
        if pred[0] == 1:
            tp += 1
        else:
            tn += 1
    else:
        wrong += 1
        if pred[0] == 1:
            fn += 1
        else:
            fp += 1
accuracy = correct/len(labels)
#print('SVC accuracy, tp, fp, tn, fn:', accuracy, tp, fp, tn, fn)
# Save svm model
dump(svc, '../svc_model.joblib') 
#clf = load('svc_model.joblib') 
#pred=[]
#for i in range(len(labels)):
#    pred.append(clf.predict([hist[i]]))

########################################## Plot ROC and AUC ######################################
# For LR
#probs = lr.predict_proba(hist)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(labels, preds, pos_label=1)
#roc_auc = metrics.auc(fpr, tpr)
#plt.figure()
#plt.title('Receiver Operating Characteristic - LR')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
#roc_auc_score(labels,preds)

## Feature importance for LR
#feat_wght_lr = np.zeros(np.shape(hist)[1]) #columns of X are features. So X.shape[1] is the number of features.
#feat_wght_lr += lr.coef_.reshape(-1) / 70
#
## For KNN
#probs = knn.predict_proba(hist)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(labels, preds, pos_label=1)
#roc_auc = metrics.auc(fpr, tpr)
#plt.figure()
#plt.title('Receiver Operating Characteristic - KNN (K=3)')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
##roc_auc_score(labels,knn_pred_labels)
#
## For SVC
#probs = svc.predict_proba(hist)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(labels, preds, pos_label=1)
#roc_auc = metrics.auc(fpr, tpr)
#plt.figure()
#plt.title('Receiver Operating Characteristic - SVC')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
##roc_auc_score(labels,svc_pred_labels)
#

############### Try with only mean of each channel #############################
def get_channel_mean(img):
    
    img = cv2.imread(img)
    size1,size2,_ = img.shape
    R = img[:,:,0].reshape(size1*size2)
    G = img[:,:,1].reshape(size1*size2)
    B = img[:,:,2].reshape(size1*size2)
    #bins = [15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255]
    
    R_mean = np.mean(R)
    G_mean = np.mean(G)
    B_mean = np.mean(B)
    
    data = [R_mean, G_mean, B_mean]
    #data = np.concatenate((R_mean,G_mean),axis=0)
    #data = np.concatenate((data,B_mean),axis=0)
    
    return data

# LOOCV - prepare data
PATH = "data//kfold"
labels = []
means = []
for file in os.listdir(PATH):
    if file[0] == 'n':
        labels.append(0)
    else:
        labels.append(1)
    means.append(get_channel_mean(os.path.join(PATH,file)))
    
# loocv with logistic regression
loocv = LeaveOneOut()
correct = 0
wrong = 0
tp = 0
fp = 0
tn = 0
fn = 0
lr_pred_labels = []
for train_idx, test_idx in loocv.split(means):
    train_mean = []
    train_labels = []
    for i in train_idx:
        train_mean.append(means[i])
        train_labels.append(labels[i])
    lr = LogisticRegression(random_state=0, solver='lbfgs').fit(train_mean, train_labels)
    pred = lr.predict([means[test_idx[0]]])
    lr_pred_labels.append(pred[0])
    if pred[0] == labels[test_idx[0]]:
        correct += 1
        if pred[0] == 1:
            tp += 1
        else:
            tn += 1
    else:
        wrong += 1
        if pred[0] == 1:
            fn += 1
        else:
            fp += 1
accuracy = correct/len(labels)
#print('LR accuracy, tp, fp, tn, fn:', accuracy, tp, fp, tn, fn)

# loocv with knn - 71.43
loocv = LeaveOneOut()
correct = 0
wrong = 0
tp = 0
fp = 0
tn = 0
fn = 0
knn_pred_labels = []
for train_idx, test_idx in loocv.split(means):
    train_mean = []
    train_labels = []
    for i in train_idx:
        train_mean.append(means[i])
        train_labels.append(labels[i])
    knn = KNeighborsClassifier(n_neighbors=3).fit(train_mean, train_labels)
    pred = knn.predict([means[test_idx[0]]])
    knn_pred_labels.append(pred[0])
    if pred[0] == labels[test_idx[0]]:
        correct += 1
        if pred[0] == 1:
            tp += 1
        else:
            tn += 1
    else:
        wrong += 1
        if pred[0] == 1:
            fn += 1
        else:
            fp += 1
accuracy = correct/len(labels)
#print('KNN accuracy, tp, fp, tn, fn:', accuracy, tp, fp, tn, fn)

# loocv with svm - 77
loocv = LeaveOneOut()
correct = 0
wrong = 0
tp = 0
fp = 0
tn = 0
fn = 0
svc_pred_labels = []
for train_idx, test_idx in loocv.split(means):
    train_mean = []
    train_labels = []
    for i in train_idx:
        train_mean.append(means[i])
        train_labels.append(labels[i])
    svc = SVC(C=2,gamma='scale',degree=3,probability=True)
    svc.fit(train_mean, train_labels)
    pred = svc.predict([means[test_idx[0]]])
    svc_pred_labels.append(pred[0])
    if pred[0] == labels[test_idx[0]]:
        correct += 1
        if pred[0] == 1:
            tp += 1
        else:
            tn += 1
    else:
        wrong += 1
        if pred[0] == 1:
            fn += 1
        else:
            fp += 1
accuracy = correct/len(labels)
#print('SVC accuracy, tp, fp, tn, fn:', accuracy, tp, fp, tn, fn)

######################## ROC and AUC
## For LR
#probs = lr.predict_proba(means)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(labels, preds, pos_label=1)
#roc_auc = metrics.auc(fpr, tpr)
#plt.figure()
#plt.title('Receiver Operating Characteristic - LR')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
##roc_auc_score(labels,preds)
#
## Feature importance for LR
#feat_wght_lr = np.zeros(np.shape(means)[1]) #columns of X are features. So X.shape[1] is the number of features.
#feat_wght_lr += lr.coef_.reshape(-1) / 70 ############## CHANGE DENOMINATOR!!!!!!
#bars = [1,2,3]
#plt.figure()
#plt.bar(bars,feat_wght_lr)
#plt.xlabel('Features - R, G, B channel means')
#plt.ylabel('Weight')
#
## For KNN
#probs = knn.predict_proba(means)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(labels, preds, pos_label=1)
#roc_auc = metrics.auc(fpr, tpr)
#plt.figure()
#plt.title('Receiver Operating Characteristic - KNN (K=3)')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
##roc_auc_score(labels,knn_pred_labels)
#
## For SVC
#probs = svc.predict_proba(means)
#preds = probs[:,1]
#fpr, tpr, threshold = metrics.roc_curve(labels, preds, pos_label=1)
#roc_auc = metrics.auc(fpr, tpr)
#plt.figure()
#plt.title('Receiver Operating Characteristic - SVC')
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0, 1])
#plt.ylim([0, 1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
#plt.show()
##roc_auc_score(labels,svc_pred_labels)









