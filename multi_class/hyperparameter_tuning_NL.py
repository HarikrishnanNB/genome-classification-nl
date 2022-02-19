#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

Hyperparameter tuning code for Neurochaos Learning
"""
import numpy as np
from load_data import pre_processing_

DATA_NAME = ['Sars_cov_2.genomes' , 'Coronaviridae.genomes', 'Metapneumovirus.genomes', 'Rhinovirus.genomes', 'Influenza.genomes' ]

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


import numpy as np
from Codes_New import  hyperparameter_tuning_bc

CLASSIFICATION_TYPE = "five_class"
EPSILON = np.arange(0.18, 0.1901, 0.0001)
INITIAL_NEURAL_ACTIVITY = np.array([0.34], dtype='float64')
DISCRIMINATION_THRESHOLD = np.array([0.499], dtype='float64')

BEST_INA, BEST_DT, BEST_EPS = hyperparameter_tuning_bc(full_genome_data, full_genome_label, CLASSIFICATION_TYPE, EPSILON, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD)
# BEST_INA = BEST INITIAL NEURAL ACTIVITY
# BEST_DT = BEST DISCRIMINATION THRESHOLD
# BEST_EPS = BEST EPSILON
