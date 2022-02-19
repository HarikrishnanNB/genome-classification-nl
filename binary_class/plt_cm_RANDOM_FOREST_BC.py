#!/usr/bin/env python3
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


## ROC Curve

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


y_score_mat = np.load(result_path + 'y_score_mat.npy')

y_test = label_binarize(true_test_label, classes=[0, 1, 2])[:,0:2]
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


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
lw = 3
plt.plot(fpr[1], tpr[1], color="red", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc[1],)
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

plt.savefig(result_path+"roc_rf_binaryclass.jpg", format='jpg', dpi=300)
plt.savefig(result_path+"roc_rf_binaryclass.eps", format='eps', dpi=300)
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