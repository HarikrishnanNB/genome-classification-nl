#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

Neurochaos Learning: Leave One Out Crossvalidator (LOOCV) Binary classification
"""
import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from load_data import binary_data_sars_1_2
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX

DATA_NAME = ['SARS-CoV-2' , 'SARS-CoV-1' ]
classification_type = 'two_class'


full_genome_data, full_genome_label = binary_data_sars_1_2()







INA = 0.34#best_initial_neural_activity[0]
D_THRESH = 0.499#best_discrimination_threshold[0]
EPSILON = 0.1830 #best_epsilon[0]

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
    
    neurochaos_train_data_features = CFX.transform(train_genome_data,
                                                               INA,
                                                               20000, EPSILON,
                                                               D_THRESH)
    neurochaos_val_data_features = CFX.transform(val_genome_data,
                                                             INA,
                                                               20000, EPSILON,
                                                               D_THRESH)

                # Neurochaos-SVM with linear kernel.

    classifier_neurochaos_svm = LinearSVC(random_state=0, tol=1e-5, dual=False)
    classifier_neurochaos_svm.fit(neurochaos_train_data_features, train_genome_label[:, 0])
    predicted_neurochaos_val_label = classifier_neurochaos_svm.predict(neurochaos_val_data_features)
    y_score = classifier_neurochaos_svm.decision_function(neurochaos_val_data_features)
    y_score_mat[ROW, :] = y_score
    print("TEST INDEX:", test_index)
    true_test_label.append(test_genome_label[0, 0])
    pred_test_label.append( predicted_neurochaos_val_label[0])
    ROW = ROW + 1

path = os.getcwd()
result_path = path + '/BC-NEUROCHAOS-RESULTS/'  + classification_type + '/CROSS_VALIDATION/'

try:
    os.makedirs(result_path)
except OSError:
    print("Creation of the result directory %s failed" % result_path)
else:
    print("Successfully created the result directory %s" % result_path)

print("Saving Hyperparameter Tuning Results")

np.save(result_path + 'true_test_label.npy', np.array(true_test_label))
np.save(result_path + 'pred_test_label.npy', np.array(pred_test_label))

np.save(result_path + 'y_score_mat.npy', y_score_mat)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_test_label, pred_test_label)
print(cm)
np.save(result_path + 'confusion_mat.npy', cm)









