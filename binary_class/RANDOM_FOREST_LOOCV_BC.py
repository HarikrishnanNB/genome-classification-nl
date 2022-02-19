#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

Random Forest: Leave One Out Crossvalidator (LOOCV) Binary classification
"""
import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from load_data import binary_data_sars_1_2

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX

DATA_NAME = ['SARS-CoV-2' , 'SARS-CoV-1' ]
classification_type = 'two_class'


full_genome_data, full_genome_label = binary_data_sars_1_2()




PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/RANDOM_FOREST/' 
    


MD = np.load(RESULT_PATH+"/h_MD.npy")[0]
NEST = np.load(RESULT_PATH+"/h_NEST.npy")[0]
F1SCORE = np.load(RESULT_PATH+"/h_F1SCORE.npy")[0]





true_test_label = []
pred_test_label = []

loo = LeaveOneOut()
loo.get_n_splits(full_genome_data)

y_score_mat = np.zeros((full_genome_data.shape[0], len(DATA_NAME)))
ROW = 0
print(loo)
LeaveOneOut()
for train_index, test_index in loo.split(full_genome_data):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print("TEST INDEX:", test_index)
    train_genome_data, val_genome_data = full_genome_data[train_index], full_genome_data[test_index]
    train_genome_label, test_genome_label = full_genome_label[train_index], full_genome_label[test_index]
    # print(X_train, X_test, y_train, y_test)
    
    clf = RandomForestClassifier( n_estimators = NEST, max_depth = MD, random_state=42)
    clf.fit(train_genome_data, train_genome_label.ravel())
    y_pred = clf.predict(val_genome_data)
    y_score = clf.predict_proba(val_genome_data)
    y_score_mat[ROW, :] = y_score
    print("TEST INDEX:", test_index)
    true_test_label.append(test_genome_label[0, 0])
    pred_test_label.append( y_pred[0])
    ROW = ROW + 1


PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/RANDOM_FOREST/' 

try:
    os.makedirs(RESULT_PATH)
except OSError:
    print("Creation of the result directory %s failed" % RESULT_PATH)
else:
    print("Successfully created the result directory %s" % RESULT_PATH)

print("Saving Hyperparameter Tuning Results")

np.save(RESULT_PATH + 'true_test_label.npy', np.array(true_test_label))
np.save(RESULT_PATH + 'pred_test_label.npy', np.array(pred_test_label))

np.save(RESULT_PATH + 'y_score_mat.npy', y_score_mat)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_test_label, pred_test_label)
print(cm)
np.save(RESULT_PATH + 'confusion_mat.npy', cm)









