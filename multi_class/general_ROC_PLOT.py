#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email: harikrishnannb07@gmail.com

ROC curve and AUC for Neurochaos Learning, SVM and Random Forest
"""
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data


DATA_NAME = ['Sars_cov_2.genomes' , 'Coronaviridae.genomes', 'Metapneumovirus.genomes', 'Rhinovirus.genomes', 'Influenza.genomes' ]

label_list = ['class-0', 'class-1', 'class-2', 'class-3', 'class-4' ]
classification_type = 'five_class'

path = os.getcwd()
general_result_path = []
result_path_nl = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/CROSS_VALIDATION/'
result_path_svm = path + '/SA-TUNING/RESULTS/SVM/' 
result_path_rf = path + '/SA-TUNING/RESULTS/RANDOM_FOREST/' 

general_result_path.append(result_path_nl)
general_result_path.append(result_path_svm)
general_result_path.append(result_path_rf)


AUC_FINAL_MICRO = []

TPR_FINAL_MICRO = []

FPR_FINAL_MICRO = []


for num_len in range(0, len(general_result_path)):
    true_test_label = np.load(general_result_path[num_len] + 'true_test_label.npy')
    pred_test_label = np.load(general_result_path[num_len] + 'pred_test_label.npy')
    
    
    
    
    pp_matrix_from_data(true_test_label, pred_test_label )
    
    cm = confusion_matrix(true_test_label, pred_test_label )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot()
    
    plt.tight_layout()
    
    # plt.savefig(result_path+"loocv_cm_chaosfex_svm.jpg", format='jpg', dpi=300)
    # plt.savefig(result_path+"loocv_cm_chaosfex_svm.eps", format='eps', dpi=300)
    
    
    ## ROC Curve
    
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    
    
    y_score_mat = np.load(general_result_path[num_len]+ 'y_score_mat.npy')
    
    y_test = label_binarize(true_test_label, classes=[0, 1, 2, 3, 4])
    n_classes = y_test.shape[1]
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score_mat[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score_mat.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    TPR_FINAL_MICRO.append(tpr["micro"])
    FPR_FINAL_MICRO.append(fpr["micro"])
    AUC_FINAL_MICRO.append(roc_auc["micro"])

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
lw = 3

plt.plot(FPR_FINAL_MICRO[0], TPR_FINAL_MICRO[0],  color="red", lw=lw, label="ROC curve for NL (AUC = %0.2f)" % 0.99 ,)
plt.plot(FPR_FINAL_MICRO[1], TPR_FINAL_MICRO[1],  color="blue", lw=lw, label="ROC curve for SVM (AUC = %0.2f)" % 0.99 ,)
plt.plot(FPR_FINAL_MICRO[2], TPR_FINAL_MICRO[2],  color="green", lw=lw, label="ROC curve for RF (AUC = %0.2f)" % 0.99 ,)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.xlabel("False Positive Rate", fontsize = 40)
plt.ylabel("True Positive Rate", fontsize = 40)
# plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right",fontsize=30)
plt.grid(True)
plt.tight_layout()

plt.savefig(result_path_nl+"roc_final_multiclass.jpg", format='jpg', dpi=300)
plt.savefig(result_path_nl+"roc_final_multiclass.eps", format='eps', dpi=300)
plt.show()


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
