from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random
import pdb
import sklearn
import os

from sklearn.metrics import roc_curve, auc
from .feature_squeezing import FeatureSqueezingDetector

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    return TP/np.sum(true_labels), FP/np.sum(1-true_labels)


def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = sklearn.metrics.accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr


from tinydb import TinyDB, Query

class DetectionEvaluator:
    """
    Get a dataset;
        Failed adv as benign / Failed adv as adversarial.
    For each detector:
        Train
        Test
        Report performance
            Detection rate on each attack.
            Detection on SAEs / FAEs.
            ROC-AUC.

    A detector should have this simplified interface:
        Y_pred = detector(X)
        
    """
    def __init__(self, model, csv_fpath):
        pass
        # set_base_model()
        self.model = model
        self.task_dir = os.path.dirname(csv_fpath)
        self.csv_fpath = csv_fpath

    def get_attack_id(self, attack_name):
        return self.attack_name_id[attack_name]

    def build_detection_dataset(self, X, Y_label, Y_pred, X_adv_list, Y_adv_pred_list, attack_names, attack_string_hash):
        # X_train, Y_train, X_test, Y_test, test_idx, failed_adv_idx = \
        #     get_detection_train_test_set(X, Y_label, X_adv_list, Y_adv_pred_list, attack_names)

        """
        Data Model:
            index, attack_id, misclassified, train
            14,     0,           0,             1
        """

        self.attack_names = attack_names
        self.attack_name_id = {}
        self.attack_name_id['legitimate'] = 0
        for i,attack_name in enumerate(attack_names):
            self.attack_name_id[attack_name] = i+1

        X_adv_all = np.concatenate(X_adv_list)
        X_leg_all = X[:len(X_adv_all)]

        self.X_detect = X_detect = np.concatenate([X_leg_all, X_adv_all])
        Y_label_adv = Y_label[:len(X_adv_list[0])]

        detection_db_path = os.path.join(self.task_dir, "detection_db_%s.json" % attack_string_hash)

        if os.path.isfile(detection_db_path):
            self.db = TinyDB(detection_db_path)
            self.query = Query()
            print ("Loaded an existing detection dataset.")
            return
        else:
            print ("Preparing the detection dataset...")

        # 1. Split Train and Test 
        random.seed(1234)
        length = len(X_detect)
        train_ratio = 0.5
        train_idx = random.sample(range(length), int(train_ratio*length))
        train_test_seq = [1 if idx in train_idx else 0 for idx in range(length) ]

        # 2. Tag the misclassified examples, both legitimate and adversarial.
        misclassified_seq = list(np.argmax(Y_label[:len(X_leg_all)], axis=1) != np.argmax(Y_pred[:len(X_leg_all)], axis=1))
        for Y_adv_pred in Y_adv_pred_list:
            misclassified_seq_adv = list(np.argmax(Y_adv_pred, axis=1) != np.argmax(Y_label_adv, axis=1))
            misclassified_seq += misclassified_seq_adv

        # 3. Tag the attack ID, 0 as legitimate.
        attack_id_seq = [0]*len(X_leg_all)
        for i,attack_name in enumerate(attack_names):
            attack_id_seq += [i+1]*len(X_adv_list[0])

        assert len(X_detect) == len(train_test_seq) == len(misclassified_seq) == len(attack_id_seq)

        self.db = TinyDB(detection_db_path)
        self.query = Query()

        for i in range(len(X_detect)):
            attack_id = attack_id_seq[i]
            misclassified = 1 if misclassified_seq[i] == True else 0
            train = train_test_seq[i]
            rec = {'index': i, 'attack_id': attack_id, 'misclassified': misclassified, 'train': train}
            self.db.insert(rec)

    def get_data_from_db_records(self, recs):
        X_idx = [rec['index'] for rec in recs]
        X = self.X_detect[np.array(X_idx)]
        Y = np.array([1 if rec['attack_id']>0 else 0 for rec in recs])
        return X, Y

    def get_training_testing_data(self, train = True):
        db = self.db
        query = self.query

        recs = db.search(query.train == 1)
        X_train, Y_train = self.get_data_from_db_records(recs)

        recs = db.search(query.train == 0)
        X_test, Y_test = self.get_data_from_db_records(recs)

        return X_train, Y_train, X_test, Y_test

    def get_sae_testing_data(self, attack_name):
        # attack_name: {[specific attack]}
        # Category: {train, test}
        # Misclassified: {True, False}
        db = self.db
        query = self.query

        attack_id = self.get_attack_id(attack_name)
        recs = db.search((query.train == 0) & (query.attack_id == attack_id) & (query.misclassified == 1))
        return self.get_data_from_db_records(recs)

    def get_detector_by_name(self, detector_name):
        model = self.model
        detector = None

        if detector_name.startswith('FeatureSqueezing'):
            detector = FeatureSqueezingDetector(model, detector_name)

        return detector

    def evaluate_detections(self, params_str):
        X_train, Y_train, X_test, Y_test = self.get_training_testing_data()

        # Example: --detection "FeatureSqueezing?distance_measure=l1&squeezers=median_smoothing_2,bit_depth_4;"
        detector_names = [ele.strip() for ele in params_str.split(';') if ele.strip()!= '']
        
        for detector_name in detector_names:
            detector = self.get_detector_by_name(detector_name)
            detector.train(X_train, Y_train)
            Y_test_pred, Y_test_pred_score = detector.test(X_test)

            accuracy, tpr, fpr = evalulate_detection_test(Y_test, Y_test_pred)
            fprs, tprs, thresholds = roc_curve(Y_test, Y_test_pred_score)
            roc_auc = auc(fprs, tprs)

            print (detector_name)
            print (accuracy, tpr, fpr, roc_auc)


            for attack_name in self.attack_names:
                X_sae, Y_sae = self.get_sae_testing_data(attack_name)
                Y_test_pred, Y_test_pred_score = detector.test(X_sae)
                _, tpr, _ = evalulate_detection_test(Y_sae, Y_test_pred)
                print ("Detection rate on SAEs: %.4f \t %s" % (tpr, attack_name))

