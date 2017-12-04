import numpy as np
from functools import reduce
import pdb

from .visualization import show_imgs_in_rows


def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]

def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]

def get_first_n_examples_id_each_class(Y_test, n=1):
    """
    Only return the classes with samples.
    """
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)

    selected_idx = []
    for i in range(num_classes):
        loc = np.where(Y_test_labels==i)[0]
        if len(loc) > 0 :
            selected_idx.append(list(loc[:n]))

    selected_idx = reduce(lambda x,y:x+y, zip(*selected_idx))

    return np.array(selected_idx)

def get_first_example_id_each_class(Y_test):
    return get_first_n_examples_id_each_class(Y_test, n=1)

def get_correct_prediction_idx(Y_pred, Y_label):
    """
    Get the index of the correct predicted samples.
    :param Y_pred: softmax output, probability matrix.
    :param Y_label: groundtruth classes in shape (#samples, #classes)
    :return: the index of samples being corrected predicted.
    """
    pred_classes = np.argmax(Y_pred, axis = 1)
    labels_classes = np.argmax(Y_label, axis = 1)

    return np.where(pred_classes == labels_classes)[0]


def calculate_mean_confidence(Y_pred, Y_target):
    """
    Calculate the mean confidence on target classes.
    :param Y_pred: softmax output
    :param Y_target: target classes in shape (#samples, #classes)
    :return: the mean confidence.
    """
    assert len(Y_pred) == len(Y_target)
    confidence = np.multiply(Y_pred, Y_target)
    confidence = np.max(confidence, axis=1)

    mean_confidence = np.mean(confidence)

    return mean_confidence

def get_match_pred_vec(Y_pred, Y_label):
    assert len(Y_pred) == len(Y_label)
    Y_pred_class = np.argmax(Y_pred, axis = 1)
    Y_label_class = np.argmax(Y_label, axis = 1)
    return Y_pred_class == Y_label_class


def calculate_accuracy(Y_pred, Y_label):
    match_pred_vec = get_match_pred_vec(Y_pred, Y_label)

    accuracy = np.sum(match_pred_vec) / float(len(Y_label))
    # pdb.set_trace()
    return accuracy


def calculate_mean_distance(X1, X2):
    img_size = X1.shape[1] * X1.shape[2]
    nb_channels = X1.shape[3]

    mean_l2_dist = np.mean([ np.sum((X1[i]-X2[i])**2)**.5 for i in range(len(X1))])
    mean_li_dist = np.mean([ np.max(np.abs(X1[i]-X2[i])) for i in range(len(X1))])
    mean_l0_dist_value = np.mean([ np.sum(X1[i]-X2[i] != 0) for i in range(len(X1))])
    mean_l0_dist_value = mean_l0_dist_value / (img_size*nb_channels)

    diff_channel_list = np.split(X1-X2 != 0, nb_channels, axis=3)
    l0_channel_dependent_list = np.sum(reduce(lambda x,y: x|y, diff_channel_list), axis = (1,2,3))
    mean_l0_dist_pixel = np.mean(l0_channel_dependent_list) / img_size

    return mean_l2_dist, mean_li_dist, mean_l0_dist_value, mean_l0_dist_pixel


def evaluate_adversarial_examples(X_test, Y_test, X_test_adv, Y_test_target, targeted, Y_test_adv_pred):
    success_rate  = calculate_accuracy(Y_test_adv_pred, Y_test_target)
    success_idx = get_match_pred_vec(Y_test_adv_pred, Y_test_target)

    if targeted is False:
        success_rate = 1 - success_rate
        success_idx = np.logical_not(success_idx)

    # Calculate the mean confidence of the successful adversarial examples.
    mean_conf = calculate_mean_confidence(Y_test_adv_pred[success_idx], Y_test_target[success_idx])
    if targeted is False:
        mean_conf = 1 - mean_conf

    mean_l2_dist, mean_li_dist, mean_l0_dist_value, mean_l0_dist_pixel = calculate_mean_distance(X_test[success_idx], X_test_adv[success_idx])
    # print ("\n---Attack: %s" % attack_string)
    print ("Success rate: %.2f%%, Mean confidence of SAEs: %.2f%%" % (success_rate*100, mean_conf*100))
    print ("### Statistics of the SAEs:")
    print ("L2 dist: %.4f, Li dist: %.4f, L0 dist_value: %.1f%%, L0 dist_pixel: %.1f%%" % (mean_l2_dist, mean_li_dist, mean_l0_dist_value*100, mean_l0_dist_pixel*100))

    rec = {}
    rec['success_rate'] = success_rate
    rec['mean_confidence'] = mean_conf
    rec['mean_l2_dist'] = mean_l2_dist
    rec['mean_li_dist'] = mean_li_dist
    rec['mean_l0_dist_value'] = mean_l0_dist_value
    rec['mean_l0_dist_pixel'] = mean_l0_dist_pixel

    return rec
