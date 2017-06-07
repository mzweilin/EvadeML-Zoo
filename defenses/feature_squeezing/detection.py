import sklearn
import numpy as np
from sklearn.metrics import roc_curve, auc
import pdb
import random

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squeeze import median_filter_np, binary_filter_np

def get_tpr_fpr(true_labels, pred, threshold):
    pred_labels = pred > threshold
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    return TP/np.sum(true_labels), FP/np.sum(1-true_labels)


class FeatureSqueezingDetector:
    def __init__(self, predict_func, squeezer_name):
        self.classifier = predict_func

        if squeezer_name == 'binary_filter':
            self.squeezer = lambda x: binary_filter_np(x)
        elif squeezer_name == 'median_filter_2':
            self.squeezer = lambda x: median_filter_np(x, 2)

    def get_l1_dist(self, X):
        pred_orig = self.classifier(X)
        pred_squeezed = self.classifier(self.squeezer(X))

        l1_dist = np.sum(np.abs(pred_orig - pred_squeezed), axis=1)
        return l1_dist

    def train(self, X, Y):
        X_l1 = self.get_l1_dist(X)

        fpr, tpr, thresholds = roc_curve(Y, X_l1)
        accuracy = [ sklearn.metrics.accuracy_score(Y, X_l1>threshold, normalize=True, sample_weight=None) for threshold in thresholds ]
        roc_auc = auc(fpr, tpr)

        idx_best = np.argmax(accuracy)
        threshold = thresholds[idx_best]
        print ("Best training accuracy: %.4f, TPR(Recall): %.4f, FPR: %.4f @%.4f" % (accuracy[idx_best], tpr[idx_best], fpr[idx_best], thresholds[idx_best]))
        print ("ROC_AUC: %.4f" % roc_auc)

        self.thresholds = thresholds
        self.threshold = threshold
        self.idx_best = idx_best

    def test(self, X, Y):
        idx_best = self.idx_best
        thresholds = self.thresholds
        threshold = thresholds[idx_best]

        X_l1 = self.get_l1_dist(X)

        accuracy_val = [ sklearn.metrics.accuracy_score(Y, X_l1>threshold, normalize=True, sample_weight=None) for threshold in thresholds ]
        tpr_val, fpr_val = zip(*[ get_tpr_fpr(Y, X_l1, threshold)  for threshold in thresholds  ])
        print ("Validation accuracy: %.4f, TPR(Recall): %.4f, FPR: %.4f @%.4f" % (accuracy_val[idx_best], tpr_val[idx_best], fpr_val[idx_best], thresholds[idx_best]))

        fpr, tpr, thresholds = roc_curve(Y, X_l1)
        roc_auc = auc(fpr, tpr)
        print ("ROC_AUC_validated: %.4f" % roc_auc)

        ret = {}
        ret['threshold'] = threshold
        ret['accuracy'] = accuracy_val[idx_best]
        ret['fpr'] = fpr_val[idx_best]
        ret['tpr'] = tpr_val[idx_best]
        ret['roc_auc'] = roc_auc

        return ret