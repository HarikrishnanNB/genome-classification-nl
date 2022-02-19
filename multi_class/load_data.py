



"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com


Data details:
    
Class-0: SARS-CoV-2 
Class-1: Coronaviradae
Class-2: Metapneumovirus
Class-3: Rhinovirus
Class-4: Influenza

Feature Transformation

"A" = 1.0
"G" = 0.75
"T" = 0.50
"C" = 0.25
"""

from Bio import SeqIO
import os
import numpy as np

import logging

from numpy.fft import fft



def pre_processing_(DATA_NAME, label, GENOME_LENGTH, SEQUENCE_THRESHOLD_LENGTH):
    """
    

    Parameters
    ----------
    DATA_NAME : list
        DATA_NAME = ['Sars_cov_2.genomes' , 'Coronaviridae.genomes', 'Metapneumovirus.genomes', 'Rhinovirus.genomes', 'Influenza.genomes' ]
    label : 1D array
        DESCRIPTION: label = [0] loads class-0 data and label
    GENOME_LENGTH : scalar (integer)
        DESCRIPTION: GENEOME_LENGTH = 8000 will consider the maximum length of genome sequence to be 8000.
    SEQUENCE_THRESHOLD_LENGTH : scalar (integer)
        DESCRIPTION: SEQUENCE_THRESHOLD_LENGTH =6000 will consider only those genome sequence having a minimum length of 6000. In other words,
        all genome sequences having length less than 6000 will not be considered for the analysis.

    Returns
    -------
    fourier_data_normalized : Numpy 2D array
        DESCRIPTION: Returns the normalized fourier pre-processed data of the respective class.
    labels : TYPE
        DESCRIPTION. Returns the labels of the fourier pre-processed data of the respective class.

    """
    
    if label[0] != 4:#'Influenza.genomes'
        for lab in label:
            num_instance = 0
            
            genome_list = []
            DATA_PATH = 'referencedata/'+ DATA_NAME[lab]+'.fasta'
            fasta_sequences = SeqIO.parse(open(DATA_PATH),'fasta')
            # with open(output_file) as out_file:
            for fasta in fasta_sequences:
                
                name, sequence = fasta.id, str(fasta.seq)
                # new_sequence = some_function(sequence)
                # write_fasta(out_file)
                
                print(name)
                if len(sequence) >= SEQUENCE_THRESHOLD_LENGTH:
                    genome_list.append(sequence)
                    num_instance = num_instance + 1
                    print("Number of samples = ", num_instance)
            genome_mat = np.zeros((num_instance, GENOME_LENGTH))
            for genome_index in range(0, num_instance):
                string = genome_list[genome_index]
                
                if len(string) < GENOME_LENGTH:
                    for num_features in range(0, len(string)):
                        if string[num_features] == 'A' or string[num_features] == 'a':
                            genome_mat[genome_index, num_features] = 1
                            
                        elif string[num_features] == 'G' or string[num_features] == 'g':
                            genome_mat[genome_index, num_features] = 0.75
                            
                        elif string[num_features] == 'T' or string[num_features] == 't':
                            genome_mat[genome_index, num_features] = 0.50
                            
                        elif string[num_features] == 'C' or string[num_features] == 'c':
                            genome_mat[genome_index, num_features] = 0.25
                        else:
                            genome_mat[genome_index, num_features] = 0
                            print(string[num_features], ", Genome index = ",genome_index, ", Feature Number = ", num_features )
                            
                else:
                    for num_features in range(0, GENOME_LENGTH):
                        
                        if string[num_features] == 'A' or string[num_features] == 'a':
                            genome_mat[genome_index, num_features] = 1
                            
                        elif string[num_features] == 'G' or string[num_features] == 'g':
                            genome_mat[genome_index, num_features] = 0.75
                            
                        elif string[num_features] == 'T' or string[num_features] == 't':
                            genome_mat[genome_index, num_features] = 0.50
                            
                        elif string[num_features] == 'C' or string[num_features] == 'c':
                            genome_mat[genome_index, num_features] = 0.25
                        else:
                            genome_mat[genome_index, num_features] = 0
                            print(string[num_features], ", Genome index = ",genome_index, ", Feature Number = ", num_features )
    
        fourier_features = np.zeros((genome_mat.shape[0], genome_mat.shape[1]))
        labels = lab * np.ones((genome_mat.shape[0], 1))
        # Computing the absolute value Fast Fourier transform coefficients of each data instance.
        for data_instance in range(0, genome_mat.shape[0]):
            fourier_features[data_instance, :] = np.abs(fft(genome_mat[data_instance, :]))
    
        # Normalization done for each row.
        numerator = fourier_features.T - np.min(fourier_features, axis=1)
        denominator = np.max(fourier_features, axis=1) - np.min(fourier_features, axis=1)
        fourier_data_normalized = (numerator/denominator).T
        # Checking whether the data is normalized.
        try:
            assert np.min(fourier_data_normalized) >= 0.0 and np.max(fourier_data_normalized) <= 1.0
        except AssertionError:
            logging.error("Error-Data should be in the range [0, 1]", exc_info=True)

        return fourier_data_normalized, labels 
    
    else:
        
        lab=label[0]
        DATA_PATH = 'referencedata/'+ DATA_NAME[lab]+'.fasta'
        fasta_sequences = SeqIO.parse(open(DATA_PATH),'fasta')
        
        skip_length=8
        
        name_list = [] 
        sequence_list = []
        for fasta in fasta_sequences:
        
            name, sequence = fasta.id, str(fasta.seq)
            name_list.append(name)
            sequence_list.append(sequence)
            
            # temp_sequence = temp_sequence+sequence
            
            # if np.mod(num, skip_length)==0:
        num_instance = np.int(len(name_list)/8)
        genome_mat = np.zeros((num_instance, GENOME_LENGTH))    
        
        string_list =[]
        temp_string = sequence_list[0]      
        for itera in range(1, len(sequence_list)):
            genome_numeric_list =[]
            string = sequence_list[itera]
            if np.mod(itera, skip_length) != 0:
                temp_string = temp_string+string
                if itera == 1023:
                    string_list.append(temp_string)
             
            else:
                string_list.append(temp_string)
                temp_string = string
            
        for genome_index in range(0, num_instance):
            string = string_list[genome_index]
            
            if len(string) < GENOME_LENGTH:
                for num_features in range(0, len(string)):
                    if string[num_features] == 'A' or string[num_features] == 'a':
                        genome_mat[genome_index, num_features] = 1
                        
                    elif string[num_features] == 'G' or string[num_features] == 'g':
                        genome_mat[genome_index, num_features] = 0.75
                        
                    elif string[num_features] == 'T' or string[num_features] == 't':
                        genome_mat[genome_index, num_features] = 0.50
                        
                    elif string[num_features] == 'C' or string[num_features] == 'c':
                        genome_mat[genome_index, num_features] = 0.25
                    else:
                        genome_mat[genome_index, num_features] = 0
                        print(string[num_features], ", Genome index = ",genome_index, ", Feature Number = ", num_features )
             
                        
         
            else:
                for num_features in range(0, GENOME_LENGTH):
                    
                    if string[num_features] == 'A' or string[num_features] == 'a':
                        genome_mat[genome_index, num_features] = 1
                        
                    elif string[num_features] == 'G' or string[num_features] == 'g':
                        genome_mat[genome_index, num_features] = 0.75
                        
                    elif string[num_features] == 'T' or string[num_features] == 't':
                        genome_mat[genome_index, num_features] = 0.50
                        
                    elif string[num_features] == 'C' or string[num_features] == 'c':
                        genome_mat[genome_index, num_features] = 0.25
                    else:
                        genome_mat[genome_index, num_features] = 0
                        print(string[num_features], ", Genome index = ",genome_index, ", Feature Number = ", num_features )    

        
                            
        fourier_features = np.zeros((genome_mat.shape[0], genome_mat.shape[1]))
        labels = lab * np.ones((genome_mat.shape[0], 1))
        # Computing the absolute value Fast Fourier transform coefficients of each data instance.
        for data_instance in range(0, genome_mat.shape[0]):
            fourier_features[data_instance, :] = np.abs(fft(genome_mat[data_instance, :]))
    
        # Normalization done for each row.
        numerator = fourier_features.T - np.min(fourier_features, axis=1)
        denominator = np.max(fourier_features, axis=1) - np.min(fourier_features, axis=1)
        fourier_data_normalized = (numerator/denominator).T
        # Checking whether the data is normalized.
        try:
            assert np.min(fourier_data_normalized) >= 0.0 and np.max(fourier_data_normalized) <= 1.0
        except AssertionError:
            logging.error("Error-Data should be in the range [0, 1]", exc_info=True)

        return fourier_data_normalized, labels 
        
        
def binary_data_sars_1_2():
    """
    

    Returns
    -------
    X_TRAIN_NORM: Returns a Numpy 2D array: Fourier preprocessed data
    LABELS: Returns label

    """
    DATA_PATH = "PREPROCESSED_DATA/" 
    COV_1_DATA = np.load(DATA_PATH + "COV_1_DATA.npy")
    COV_1_LABEL = np.load(DATA_PATH + "COV_1_LABEL.npy")

    COV_2_DATA = np.load(DATA_PATH + "COV_2_DATA.npy")
    COV_2_LABEL = np.load(DATA_PATH + "COV_2_LABEL.npy")
    
    DATA = np.vstack((COV_1_DATA, COV_2_DATA))
    LABELS = np.vstack((COV_1_LABEL,COV_2_LABEL))
    
    FOURIER_FEATURES = np.zeros((DATA.shape[0],DATA.shape[1]))
    for data_instance in range(0, DATA.shape[0]):
        FOURIER_FEATURES[data_instance,:]  = np.abs( fft( DATA[data_instance,:] ) )

    X_TRAIN_NORM = ((FOURIER_FEATURES.T - np.min(FOURIER_FEATURES, axis = 1))/(np.max(FOURIER_FEATURES, axis= 1) - np.min(FOURIER_FEATURES, axis = 1))).T




    try:
        assert np.min(X_TRAIN_NORM) >= 0.0 and np.max(X_TRAIN_NORM) <= 1.0
    except AssertionError:
        logging.error("Train Data is NOT normalized. Hint: Go to get_data() function and normalize the data to lie in the range [0, 1]", exc_info=True)

#      
    return X_TRAIN_NORM, LABELS

