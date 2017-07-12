import numpy as np
import functools
import pdb

def get_first_example_id_each_class(Y_test):
    """
    Only return the classes with samples.
    """
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)

    selected_idx = []
    for i in range(num_classes):
        loc = np.where(Y_test_labels==i)[0]
        if len(loc) > 0 :
            selected_idx.append(loc[0])

    return selected_idx


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


def calculate_accuracy(Y_pred, Y_label):
    assert len(Y_pred) == len(Y_label)
    Y_pred_class = np.argmax(Y_pred, axis = 1)
    Y_label_class = np.argmax(Y_label, axis = 1)

    accuracy = np.sum(Y_pred_class == Y_label_class) / float(len(Y_label))
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
    l0_channel_dependent_list = np.sum(functools.reduce(lambda x,y: x|y, diff_channel_list), axis = (1,2,3))
    mean_l0_dist_pixel = np.mean(l0_channel_dependent_list) / img_size

    return mean_l2_dist, mean_li_dist, mean_l0_dist_value, mean_l0_dist_pixel


def evaluate_adversarial_examples(X_test, X_test_adv, Y_test_target, targeted, Y_test_adv_pred):
    success_rate = calculate_accuracy(Y_test_adv_pred, Y_test_target)
    # TODO: calculate the mean confidence of the successful adversarial examples.
    mean_conf = calculate_mean_confidence(Y_test_adv_pred, Y_test_target)
    if targeted is False:
        success_rate = 1 - success_rate
        mean_conf = 1 - mean_conf

    mean_l2_dist, mean_li_dist, mean_l0_dist_value, mean_l0_dist_pixel = calculate_mean_distance(X_test, X_test_adv)

    # print ("\n---Attack: %s" % attack_string)
    print ("Success rate: %.2f%%, Mean confidence: %.2f%%" % (success_rate*100, mean_conf*100))
    print ("L2 dist: %.4f, Li dist: %.4f, L0 dist_value: %.1f%%, L0 dist_pixel: %.1f%%" % (mean_l2_dist, mean_li_dist, mean_l0_dist_value*100, mean_l0_dist_pixel*100))
    # print ("Duration: %.4f per sample" % dur_per_sample)

    rec = {}
    rec['success_rate'] = success_rate
    rec['mean_confidence'] = mean_conf
    rec['mean_l2_dist'] = mean_l2_dist
    rec['mean_li_dist'] = mean_li_dist
    rec['mean_l0_dist_value'] = mean_l0_dist_value
    rec['mean_l0_dist_pixel'] = mean_l0_dist_pixel
    
    return rec