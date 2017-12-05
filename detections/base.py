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
from .magnet_mnist import MagNetDetector as MagNetDetectorMNIST
from .magnet_cifar import MagNetDetector as MagNetDetectorCIFAR

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS
from utils.output import write_to_csv

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP


def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = sklearn.metrics.accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr, tp, ap = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr, tp, ap


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
    def __init__(self, model, result_folder, csv_fname, dataset_name):
        pass
        # set_base_model()
        self.model = model
        self.task_dir = result_folder
        self.csv_fpath = os.path.join(result_folder, csv_fname)
        self.dataset_name = dataset_name

        if not os.path.isdir(self.task_dir):
            os.makedirs(self.task_dir)

    def get_attack_id(self, attack_name):
        return self.attack_name_id[attack_name]

    def build_detection_dataset(self, X, Y_label, Y_pred, selected_idx, X_adv_list, Y_adv_pred_list, attack_names, attack_string_hash, clip, Y_test_target_next, Y_test_target_ll):
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
        # TODO: this could be wrong in non-default data selection mode.
        Y_label_adv = Y_label[selected_idx]

        detection_db_path = os.path.join(self.task_dir, "detection_db_%s_clip_%s.json" % (attack_string_hash, clip))

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
        # TODO: Differentiate the successful examples between targeted and non-targeted.
        misclassified_seq = list(np.argmax(Y_label[:len(X_leg_all)], axis=1) != np.argmax(Y_pred[:len(X_leg_all)], axis=1))
        for Y_adv_pred in Y_adv_pred_list:
            misclassified_seq_adv = list(np.argmax(Y_adv_pred, axis=1) != np.argmax(Y_label_adv, axis=1))
            misclassified_seq += misclassified_seq_adv

        success_adv_seq = [False] * len(X_leg_all)
        for i, Y_adv_pred in enumerate(Y_adv_pred_list):
            attack_name = attack_names[i]
            if 'targeted=ll' in attack_name:
                success_adv_seq_attack = list(np.argmax(Y_adv_pred, axis=1) == np.argmax(Y_test_target_ll, axis=1))
            elif 'targeted=next' in attack_name:
                success_adv_seq_attack = list(np.argmax(Y_adv_pred, axis=1) == np.argmax(Y_test_target_next, axis=1))
            else:
                # The same as misclassified.
                success_adv_seq_attack = list(np.argmax(Y_adv_pred, axis=1) != np.argmax(Y_label_adv, axis=1))
            success_adv_seq += success_adv_seq_attack



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
            success = 1 if success_adv_seq[i] == True else 0
            train = train_test_seq[i]
            rec = {'index': i, 'attack_id': attack_id, 'misclassified': misclassified, 'success': success, 'train': train}
            self.db.insert(rec)

    def get_data_from_db_records(self, recs):
        if len(recs) == 0:
            return None, None
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

    def get_adversarial_data(self, only_testing, success, attack_name=None, include_legitimate=False):
        db = self.db
        query = self.query

        conditions_and = []
        if only_testing:
            conditions_and.append(query.train == 0)

        if attack_name is None:
            conditions_and.append(query.attack_id > 0)
        else:
            attack_id = self.get_attack_id(attack_name)
            conditions_and.append(query.attack_id == attack_id)

        if success:
            conditions_and.append(query.success == 1)
        else:
            conditions_and.append(query.success == 0)

        conditions = reduce(lambda a,b:a&b, conditions_and)
        # print ("conditions: %s " % conditions)

        recs = db.search(conditions)

        if include_legitimate:
            if only_testing:
                conditions = (query.attack_id == 0) & (query.train == 0)
            else:
                conditions = query.attack_id == 0
            # print ("additional conditions: %s " % conditions)
            recs += db.search(conditions)

        return self.get_data_from_db_records(recs)

    def get_sae_testing_data(self, attack_name=None):
        return self.get_adversarial_data(only_testing=True, success=True, attack_name=attack_name)

    def get_sae_data(self, attack_name=None):
        return self.get_adversarial_data(only_testing=False, success=True, attack_name=attack_name)

    def get_fae_testing_data(self, attack_name=None):
        return self.get_adversarial_data(only_testing=True, success=False, attack_name=attack_name)

    def get_fae_data(self, attack_name=None):
        return self.get_adversarial_data(only_testing=False, success=False, attack_name=attack_name)

    def get_all_non_fae_testing_data(self, attack_name=None):
        return self.get_adversarial_data(only_testing=True, success=True, attack_name=attack_name, include_legitimate=True)

    def get_all_non_fae_data(self, attack_name=None):
        return self.get_adversarial_data(only_testing=False, success=True, attack_name=attack_name, include_legitimate=True)

    def get_detector_by_name(self, detector_name):
        model = self.model
        detector = None

        if detector_name.startswith('FeatureSqueezing'):
            detector = FeatureSqueezingDetector(model, detector_name)
        elif detector_name.startswith('MagNet'):
            if self.dataset_name == 'MNIST':
                detector = MagNetDetectorMNIST(model, detector_name)
            elif self.dataset_name == "CIFAR-10":
                detector = MagNetDetectorCIFAR(model, detector_name)

        return detector

    def evaluate_detections(self, params_str):
        X_train, Y_train, X_test, Y_test = self.get_training_testing_data()

        # Example: --detection "FeatureSqueezing?distance_measure=l1&squeezers=median_smoothing_2,bit_depth_4;"
        detector_names = [ele.strip() for ele in params_str.split(';') if ele.strip()!= '']

        dataset_name = self.dataset_name
        csv_fpath = "./detection_%s_saes.csv" % dataset_name
        fieldnames = ['detector', 'threshold', 'fpr'] + self.attack_names + ['overall']
        to_csv = []

        for detector_name in detector_names:
            detector = self.get_detector_by_name(detector_name)
            if detector is None:
                print ("Skipped an unknown detector [%s]" % detector_name.split('?')[0])
                continue
            detector.train(X_train, Y_train)
            Y_test_pred, Y_test_pred_score = detector.test(X_test)

            accuracy, tpr, fpr, tp, ap = evalulate_detection_test(Y_test, Y_test_pred)
            fprs, tprs, thresholds = roc_curve(Y_test, Y_test_pred_score)
            roc_auc = auc(fprs, tprs)

            print ("Detector: %s" % detector_name)
            print ("Accuracy: %f\tTPR: %f\tFPR: %f\tROC-AUC: %f" % (accuracy, tpr, fpr, roc_auc))

            rec = {}
            rec['detector'] = detector_name
            if hasattr(detector, 'threshold'):
                rec['threshold'] = detector.threshold
            else:
                rec['threshold'] = None
            rec['fpr'] = fpr
            overall_detection_rate_saes = 0
            nb_saes = 0
            for attack_name in self.attack_names:
                # No adversarial examples for training for the current detection methods.
                # X_sae, Y_sae = self.get_sae_testing_data(attack_name)
                if FLAGS.detection_train_test_mode:
                    X_sae, Y_sae = self.get_sae_testing_data(attack_name)
                else:
                    X_sae, Y_sae = self.get_sae_data(attack_name)
                Y_test_pred, Y_test_pred_score = detector.test(X_sae)
                _, tpr, _, tp, ap = evalulate_detection_test(Y_sae, Y_test_pred)
                print ("Detection rate on SAEs: %.4f \t %3d/%3d \t %s" % (tpr, tp, ap, attack_name))
                overall_detection_rate_saes += tpr * len(Y_sae)
                nb_saes += len(Y_sae)
                rec[attack_name] = tpr
                # print ("overall_detection_rate_saes/nb_saes: %d/%d" % (overall_detection_rate_saes, nb_saes))

            print ("Overall detection rate on SAEs: %f (%d/%d)" % (overall_detection_rate_saes/nb_saes, overall_detection_rate_saes, nb_saes))
            rec['overall'] = float(overall_detection_rate_saes/nb_saes)
            to_csv.append(rec)

            # No adversarial examples for training for the current detection methods.
            # X_sae_all, Y_sae_all = self.get_sae_testing_data()
            print ("### Excluding FAEs:")
            if FLAGS.detection_train_test_mode:
                X_nfae_all, Y_nfae_all = self.get_all_non_fae_testing_data()
            else:
                X_nfae_all, Y_nfae_all = self.get_all_non_fae_data()
            Y_pred, Y_pred_score = detector.test(X_nfae_all)
            _, tpr, _, tp, ap = evalulate_detection_test(Y_nfae_all, Y_pred)
            fprs, tprs, thresholds = roc_curve(Y_nfae_all, Y_pred_score)

            # print ("threshold\tfpr\ttpr")
            # for i, threshold  in enumerate(thresholds):
            #     print ("%.4f\t%.4f\t%.4f" % (threshold, fprs[i], tprs[i]))

            roc_auc = auc(fprs, tprs)
            print ("Overall TPR: %f\tROC-AUC: %f" % (tpr, roc_auc))

            # FAEs
            if FLAGS.detection_train_test_mode:
                X_fae, Y_fae = self.get_fae_testing_data()
            else:
                X_fae, Y_fae = self.get_fae_data()
            Y_test_pred, Y_test_pred_score = detector.test(X_fae)
            _, tpr, _, tp, ap = evalulate_detection_test(Y_fae, Y_test_pred)
            print ("Overall detection rate on FAEs: %.4f \t %3d/%3d" % (tpr, tp, ap))

        write_to_csv(to_csv, csv_fpath, fieldnames)