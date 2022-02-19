#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

Neurochaos Learning: Leave One Out Crossvalidator (LOOCV)
"""
import os
import numpy as np
from sklearn.model_selection import LeaveOneOut
from load_data import pre_processing_
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX

DATA_NAME = ['Sars_cov_2.genomes' , 'Coronaviridae.genomes', 'Metapneumovirus.genomes', 'Rhinovirus.genomes', 'Influenza.genomes' ]
classification_type = 'five_class'


GENOME_LENGTH = 8000
SEQUENCE_THRESHOLD_LENGTH = 6000  
label = [0]      
class_0_data, class_0_label  = pre_processing_(DATA_NAME, label, GENOME_LENGTH, SEQUENCE_THRESHOLD_LENGTH)
label = [1]
class_1_data, class_1_label  = pre_processing_(DATA_NAME, label, GENOME_LENGTH, SEQUENCE_THRESHOLD_LENGTH)
label = [2]
class_2_data, class_2_label  = pre_processing_(DATA_NAME, label, GENOME_LENGTH, SEQUENCE_THRESHOLD_LENGTH)
label = [3]
class_3_data, class_3_label  = pre_processing_(DATA_NAME, label, GENOME_LENGTH, SEQUENCE_THRESHOLD_LENGTH)
label = [4]
class_4_data, class_4_label  = pre_processing_(DATA_NAME, label, GENOME_LENGTH, SEQUENCE_THRESHOLD_LENGTH)
    

full_genome_data = np.concatenate((class_0_data, class_1_data, class_2_data, class_3_data, class_4_data))

full_genome_label = np.concatenate((class_0_label, class_1_label, class_2_label, class_3_label, class_4_label))




path = os.getcwd()
result_path_hyper = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/CROSS_VALIDATION/'

f1score_matrix = np.load(result_path_hyper + 'H_FSCORE.npy')
q_matrix =  np.load(result_path_hyper + 'H_INITIAL_CONDITION.npy')
epsilon_matrix = np.load(result_path_hyper + 'H_EPS.npy')
b_matrix = np.load(result_path_hyper + 'H_THRESHOLD.npy')


maximum_fscore = np.max(f1score_matrix)

best_initial_neural_activity = []
best_discrimination_threshold = []
best_epsilon = []
for row in range(0, f1score_matrix.shape[0]):

    for col in range(0, f1score_matrix.shape[1]):

        if f1score_matrix[row, col] == np.max(f1score_matrix):

            best_initial_neural_activity.append(q_matrix[row, col])
            best_discrimination_threshold.append(b_matrix[row, col])
            best_epsilon.append(epsilon_matrix[row, col])

print("maximum f1score_neurochaos = ", maximum_fscore)
print("best initial neural activity = ", best_initial_neural_activity)
print("best discrimination threshold = ", best_discrimination_threshold)
print("best epsilon = ", best_epsilon)



INA = best_initial_neural_activity[0]
D_THRESH = best_discrimination_threshold[0]
EPSILON = best_epsilon[0]

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

result_path = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/CROSS_VALIDATION/'

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









