from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import calculate_accuracy

class RobustClassifierBase:
    def __init__(self, model, rc_name):
        self.model_predict = lambda x: model.predict(x)

    def predict(self, X):
        return self.model_predict(X)

from .feature_squeezing import FeatureSqueezingRC

def get_robust_classifier_by_name(model, rc_name):
    if rc_name.startswith('Base'):
        rc = RobustClassifierBase(model, rc_name)
    elif rc_name.startswith("FeatureSqueezing"):
        rc = FeatureSqueezingRC(model, rc_name)
    return rc

def evaluate_robustness(params_str, model, Y, X, attack_string_list, X_adv_list, csv_fpath):
    RC_names = [ele.strip() for ele in params_str.split(';') if ele.strip()!= '']
    Y_adv = Y[:len(X_adv_list[0])]

    accuracy_rows = []
    fieldnames = ['RobustClassifier', 'legitimate_%d' % len(X)] + attack_string_list

    for RC_name in RC_names:
        rc = get_robust_classifier_by_name(model, RC_name)
        accuracy_rec = {}
        accuracy_rec['RobustClassifier'] = RC_name

        accuracy = calculate_accuracy(rc.predict(X), Y)
        accuracy_rec['legitimate_%d' % len(X)] = accuracy

        for i, attack_name in enumerate(attack_string_list):
            X_adv = X_adv_list[i]
            Y_pred_adv = rc.predict(X_adv)
            accuracy = calculate_accuracy(Y_pred_adv, Y_adv)
            accuracy_rec[attack_name] = accuracy        

        accuracy_rows.append(accuracy_rec)

    # Output in a CSV file.
    import csv
    with open(csv_fpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in accuracy_rows:
            writer.writerow(row)

