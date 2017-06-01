import numpy as np
import pdb

def get_first_example_id_each_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    selected_idx = [ np.where(Y_test_labels==i)[0][0] for i in range(num_classes)]
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
    return accuracy


def calculate_mean_distance(X1, X2):
    mean_l2_dist = np.mean([ np.sum((X1[i]-X2[i])**2)**.5 for i in range(len(X1))])
    mean_li_dist = np.mean([ np.max(np.abs(X1[i]-X2[i])) for i in range(len(X1))])
    mean_l0_dist = np.mean([ np.sum(X1[i]-X2[i] != 0) for i in range(len(X1))])
    return mean_l2_dist, mean_li_dist, mean_l0_dist