#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

Hyperparameter Tuning Random Forest
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split




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

X_train_norm, X_test_norm, y_train, y_test = train_test_split(full_genome_data, full_genome_label, test_size=0.2, random_state=42)

#Algorithm - Random Forest
n_estimator = [1, 10, 100, 1000, 10000]
BESTF1 = 0
FOLD_NO = 5
KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True)  
KF.get_n_splits(X_train_norm) 
print(KF) 
for NEST in n_estimator:
                
    for MD in range(1,11):


        FSCORE_TEMP=[]
    
        for TRAIN_INDEX, VAL_INDEX in KF.split(X_train_norm):
            
            X_TRAIN, X_VAL = X_train_norm[TRAIN_INDEX], X_train_norm[VAL_INDEX]
            Y_TRAIN, Y_VAL = y_train[TRAIN_INDEX], y_train[VAL_INDEX]
        
            
            clf = RandomForestClassifier( n_estimators = NEST, max_depth = MD, random_state=42)
            clf.fit(X_TRAIN, Y_TRAIN.ravel())
            Y_PRED = clf.predict(X_VAL)
            f1 = f1_score(Y_VAL, Y_PRED, average='macro')
            FSCORE_TEMP.append(f1)
            print('F1 Score', f1)
        print("Mean F1-Score for N-EST = ", NEST," MD = ", MD," is  = ",  np.mean(FSCORE_TEMP)  )
        if(np.mean(FSCORE_TEMP) > BESTF1):
            BESTF1 = np.mean(FSCORE_TEMP)
            BESTNEST = NEST
            BESTMD = MD
            

print("BEST F1SCORE", BESTF1)
print("BEST MD = ", BESTMD)
print("BEST NEST = ", BESTNEST)




print("Saving Hyperparameter Tuning Results")
   
  
PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/RANDOM_FOREST/'


try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/h_NEST.npy", np.array([BESTNEST]) ) 
np.save(RESULT_PATH+"/h_MD.npy", np.array([BESTMD]) ) 
np.save(RESULT_PATH+"/h_F1SCORE.npy", np.array([BESTF1]) ) 
