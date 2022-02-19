#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

Hyperparameter tuning code for NL for the binary classification problem
"""


import numpy as np
from load_data import binary_data_sars_1_2

import numpy as np
from Codes_New import  hyperparameter_tuning_bc



CLASSIFICATION_TYPE = "two_class"
full_genome_data, full_genome_label = binary_data_sars_1_2()
EPSILON = np.arange(0.18, 0.1901, 0.0001)
INITIAL_NEURAL_ACTIVITY = np.array([0.34], dtype='float64')
DISCRIMINATION_THRESHOLD = np.array([0.499], dtype='float64')

BEST_INA, BEST_DT, BEST_EPS = hyperparameter_tuning_bc(full_genome_data, full_genome_label, CLASSIFICATION_TYPE, EPSILON, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD)
# BEST_INA = BEST INITIAL NEURAL ACTIVITY
# BEST_DT = BEST DISCRIMINATION THRESHOLD
# BEST_EPS = BEST EPSILON
