from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import hashlib
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import calculate_accuracy
from datasets.visualization import show_imgs_in_rows


class RobustClassifierBase:
    def __init__(self, model, rc_name):
        self.model_predict = lambda x: model.predict(x)

    def predict(self, X):
        return self.model_predict(X)

from .feature_squeezing import FeatureSqueezingRC
from .magnet import MagNetRC

def get_robust_classifier_by_name(model, rc_name):
    if rc_name.startswith('Base') or rc_name.startswith('none'):
        rc = RobustClassifierBase(model, rc_name)
    elif rc_name.startswith("FeatureSqueezing"):
        rc = FeatureSqueezingRC(model, rc_name)
    elif rc_name.startswith("MagNet"):
        rc = MagNetRC(model, rc_name)
    else:
        raise Exception("Unknown robust classifier [%s]" % rc)
    return rc

def evaluate_robustness(params_str, model, Y, X, Y_adv, attack_string_list, X_adv_list, fname_prefix, selected_idx_vis, result_folder):
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)
    robustness_string_hash = hashlib.sha1(params_str.encode('utf-8')).hexdigest()[:5]
    csv_fname = "%s_%s.csv" % (fname_prefix, robustness_string_hash)
    csv_fpath = os.path.join(result_folder, csv_fname)
    print ("Saving robustness test results at %s" % csv_fpath)

    RC_names = [ele.strip() for ele in params_str.split(';') if ele.strip()!= '']

    accuracy_rows = []
    fieldnames = ['RobustClassifier', 'legitimate_%d' % len(X)] + attack_string_list

    selected_idx_vis = selected_idx_vis[:10]
    legitimate_examples = X[selected_idx_vis]

    for RC_name in RC_names:
        rc = get_robust_classifier_by_name(model, RC_name)
        accuracy_rec = {}
        accuracy_rec['RobustClassifier'] = RC_name

        accuracy = calculate_accuracy(rc.predict(X), Y)
        accuracy_rec['legitimate_%d' % len(X)] = accuracy

        img_fpath = os.path.join(result_folder, '%s_%s.png' % (fname_prefix, RC_name) )
        rows = [legitimate_examples]

        for i, attack_name in enumerate(attack_string_list):
            X_adv = X_adv_list[i]
            if hasattr(rc, 'visualize_and_predict'):
                X_adv_filtered, Y_pred_adv = rc.visualize_and_predict(X_adv)
                rows += map(lambda x:x[selected_idx_vis], [X_adv, X_adv_filtered])
            else:
                Y_pred_adv = rc.predict(X_adv)
            accuracy = calculate_accuracy(Y_pred_adv, Y_adv)
            accuracy_rec[attack_name] = accuracy        

        accuracy_rows.append(accuracy_rec)

        # Visualize the filtered images.
        if len(rows) > 1:
            show_imgs_in_rows(rows, img_fpath)

    # Output in a CSV file.
    import csv
    with open(csv_fpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in accuracy_rows:
            writer.writerow(row)

