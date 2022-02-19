#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:23:38 2022

@author: harikrishnan
"""




import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX

def chaosnet(traindata, trainlabel, testdata):
    '''
    

    Parameters
    ----------
    traindata : TYPE - Numpy 2D array
        DESCRIPTION - traindata
    trainlabel : TYPE - Numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE - Numpy 2D array
        DESCRIPTION - testdata

    Returns
    -------
    mean_each_class : Numpy 2D array
        DESCRIPTION - mean representation vector of each class
    predicted_label : TYPE - numpy 1D array
        DESCRIPTION - predicted label

    '''
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:,0], :], axis=0)
        
    predicted_label = np.argmax(cosine_similarity(testdata, mean_each_class), axis = 1)

    return mean_each_class, predicted_label



def hyperparameter_tuning_bc(full_genome_data, full_genome_label, classification_type, epsilon, initial_neural_activity, discrimination_threshold):
    """
    This module does hyperparameter tuning to find the best epsilon,
    and discrimination threshold.
    Parameters
    ----------
    classification_type : string
        DESCRIPTION- classification_type = "binary_class", loads binary
        classification data.
        classification_type = "multi_class", loads multiclass classification
        data.
    epsilon : array, 1D
        DESCRIPTION - epsilon - is the neighbourhood of the stimulus. Epsilon is
        a value between 0 and 0.3.

    initial_neural_activity : array, 1D (This array should contain
    only one element, for eg. np.array([0.34],dtype = 'float64')).

        DESCRIPTION - Every chaotic neuron has an initial neural activity.
        The firinig of chaotic neuron starts from this value.

    discrimination_threshold : array, 1D
        DESCRIPTION - discrimination threshold is used to calculate the
        fraction of time the chaotic trajrectory is above this threshold.
        For more informtion, refer the following: https://aip.scitation.org/doi/abs/10.1063/1.5120831?journalCode=cha

    Returns
    -------
    best_initial_neural_activity : array, 1D (This array has only one element).
        DESCRIPTION - return initial neural activity
    best_discrimination_threshold : array, 1D
        DESCRIPTION - return discrimniation threshold
    best_epsilon : array, 1D
        DESCRIPTION - return best epsilon

    """
  
    accuracy_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    f1score_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    q_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    b_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))
    epsilon_matrix = np.zeros((len(discrimination_threshold), len(epsilon)))

    # Define the split - into 2 folds
    k_fold = KFold(n_splits=3, random_state=42, shuffle=True)

    # returns the number of splitting iterations in the cross-validator
    k_fold.get_n_splits(full_genome_data)
    print(k_fold)

    KFold(n_splits=3, random_state=42, shuffle=True)
    row = -1
    col = -1

    initial_condition_instance = initial_neural_activity[0]
    for threshold_instance in discrimination_threshold:
        row = row+1
        col = -1

        for epsilon_instance in epsilon:
            col = col+1
            acc_temp = []
            fscore_temp = []

            for train_index, val_index in k_fold.split(full_genome_data):

                train_genome_data = full_genome_data[train_index]
                val_genome_data = full_genome_data[val_index]
                train_genome_label = full_genome_label[train_index]
                val_genome_label = full_genome_label[val_index]
                print(" train data (%) = ",
                      (train_genome_data.shape[0]/full_genome_data.shape[0])*100)
                print("val data (%) = ",
                      (val_genome_data.shape[0]/full_genome_data.shape[0])*100)


                neurochaos_train_data_features = CFX.transform(train_genome_data,
                                                               initial_condition_instance,
                                                               20000, epsilon_instance,
                                                               threshold_instance)
                neurochaos_val_data_features = CFX.transform(val_genome_data,
                                                             initial_condition_instance,
                                                             20000, epsilon_instance,
                                                             threshold_instance)

                # Neurochaos-SVM with linear kernel.

                classifier_neurochaos_svm = LinearSVC(random_state=0, tol=1e-5, dual=False)
                classifier_neurochaos_svm.fit(neurochaos_train_data_features,
                                              train_genome_label[:, 0])
                predicted_neurochaos_val_label = classifier_neurochaos_svm.predict(neurochaos_val_data_features)
                # Accuracy
                acc_neurochaos = accuracy_score(val_genome_label, predicted_neurochaos_val_label)*100
                # Macro F1- Score
                f1score_neurochaos = f1_score(val_genome_label, predicted_neurochaos_val_label, average="macro")

                acc_temp.append(acc_neurochaos)
                fscore_temp.append(f1score_neurochaos)

            q_matrix[row, col] = initial_condition_instance
            b_matrix[row, col] = threshold_instance
            epsilon_matrix[row, col] = epsilon_instance
            # Average Accuracy
            accuracy_matrix[row, col] = np.mean(acc_temp)
            # Average Macro F1-score
            f1score_matrix[row, col] = np.mean(fscore_temp)

            print("q_matrix = ", q_matrix[row, col],
                  "b_matrix = ", b_matrix[row, col],
                  "epsilon = ", epsilon_matrix[row, col])
            print("Three fold Average F-SCORE %.3f" %f1score_matrix[row, col])

            print('--------------------------')
    # Creating a result path to save the results.
    path = os.getcwd()
    result_path = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/CROSS_VALIDATION/'


    try:
        os.makedirs(result_path)
    except OSError:
        print("Creation of the result directory %s failed" % result_path)
    else:
        print("Successfully created the result directory %s" % result_path)

    print("Saving Hyperparameter Tuning Results")
    np.save(result_path + 'H_ACCURACY.npy', accuracy_matrix)
    np.save(result_path + 'H_FSCORE.npy', f1score_matrix)
    np.save(result_path + 'H_INITIAL_CONDITION.npy', q_matrix)
    np.save(result_path + 'H_THRESHOLD.npy', b_matrix)
    np.save(result_path + 'H_EPS.npy', epsilon_matrix)
    # =============================================================================
    # best hyperparameters
    # =============================================================================
    # Computing the maximum F1-score obtained during crossvalidation.
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

    return best_initial_neural_activity, best_discrimination_threshold, best_epsilon

def classification_report_csv_(report, num_classes):
    """
    This module returns the classfication metric for binary classification and
    five class classification as a dataframe. The module currently works for
    binary and five class classification.
    Parameters
    ----------
    report : classification metric report
        DESCRIPTION - Contains the precision recall f1score
    num_classes : int
        DESCRIPTION - 2 or 5, if 2 the report for binary class is returned
        if 5 the report for 5 class classification is returned.
    Returns
    -------
    dataframe - contains precision recall f1score

    """
    if num_classes == 2:
        report_data = []
        lines = report.split('\n')
        report_data.append(lines[0])
        report_data.append(lines[2])
        report_data.append(lines[3])
        report_data.append(lines[5])
        report_data.append(lines[6])
        report_data.append(lines[7])
        dataframe = pd.DataFrame.from_dict(report_data)
    #    dataframe.to_csv('report.csv', index = False)
        return dataframe
    elif num_classes == 5:
        report_data = []
        lines = report.split('\n')
        report_data.append(lines[0])
        report_data.append(lines[2])
        report_data.append(lines[3])
        report_data.append(lines[4])
        report_data.append(lines[5])
        report_data.append(lines[6])
        report_data.append(lines[8])
        report_data.append(lines[9])
        report_data.append(lines[10])
        dataframe = pd.DataFrame.from_dict(report_data)
    #    dataframe.to_csv('report.csv', index = False)
        return dataframe
