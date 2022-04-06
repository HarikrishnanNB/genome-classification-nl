# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

Confusion Matrix plot for Random Forest for LOOCV
"""
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data

DATA_NAME = ['SARS-CoV-2' , 'SARS-CoV-1' ]
classification_type = 'two_class'


label_list = ['class-0', 'class-1']


path = os.getcwd()
result_path = path + '/SA-TUNING/RESULTS/RANDOM_FOREST/' 

true_test_label = np.load(result_path + 'true_test_label.npy')
pred_test_label = np.load(result_path + 'pred_test_label.npy')




pp_matrix_from_data(true_test_label, pred_test_label )

cm = confusion_matrix(true_test_label, pred_test_label )
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
disp.plot()
plt.tight_layout()

plt.savefig(result_path+"/loocv_cm_rf_binaryclass.jpg", format='jpg', dpi=300)
plt.savefig(result_path+"loocv_cm_rf_binaryclass.eps", format='eps', dpi=300)


# Classification Metrics
from sklearn.metrics import classification_report
print(classification_report(true_test_label, pred_test_label , target_names=label_list))

from sklearn.metrics import precision_score

precision = precision_score(true_test_label, pred_test_label, average='macro')

from sklearn.metrics import recall_score
recall = recall_score(true_test_label, pred_test_label, average='macro')

from sklearn.metrics import f1_score
fscore = f1_score(true_test_label, pred_test_label, average='macro')

print("Precision = ", precision)
print("Recall = ", recall)
print("F1-score = ", fscore)



classification_metric = np.zeros((10, cm.shape[1]+1), dtype=object)
classification_metric[0, :] = ['metric', 'class_0', 'class_1']
classification_metric[1:, 0] = ['SE','SP', 'ACC', 'PPV', 'NPV', 'FPR', 'FDR', 'FNR', 'F1']
COLUMN_INDEX=0

for label in range(0, cm.shape[0]):
    COLUMN_INDEX = COLUMN_INDEX+1
    
    TP = cm[label, label]
    TN = 0
    FP = 0
    FN = 0
    
    for row in range(0, cm.shape[0]):
        
        # for col in range(0, cm_1.shape[1]):
        
        for col in range(0, cm.shape[1]):
            if row!=label and col!=label:
                TN = TN+cm[row, col]
            if row == label and col != label:
                FN = FN + cm[row, col]
                
            if row !=label and col == label:
                FP = FP +cm[row, label]
    # print("Class-", label)
    # print("-----------")          
    # print("TP = ", TP)
    # print("TN = ", TN)
    # print("FP = ", FP)
    # print("FN = ", FN)
    # print("***********")
    # Sensitivity
    SE = TP/(TP+FN)
    # Specificity
    SP = TN/(TN+FP)
    # Accuracy
    ACC = (TP+TN)/(TP+TN+FP+FN)
    print(ACC)
    # PPV
    PPV = TP/(TP+FP)
    # NPV
    NPV = TN/(TN+FN)
    # FPR
    FPR = FP/(FP+TN)
    # FDR
    FDR = FP/(FP+TP)
    # FNR
    FNR = FN/(FN+TP)
    # F1-score
    F1_ = 2*PPV*SE/(PPV+SE)
    
    result_array = [SE, SP, ACC, PPV, NPV, FPR, FDR, FNR, F1_]
    classification_metric[1:, COLUMN_INDEX] =  result_array
print("Classification Metric= \n", classification_metric)
print("Average Scores",np.mean(classification_metric[1:,1:],1))