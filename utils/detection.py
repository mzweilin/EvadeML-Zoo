import numpy as np
import random
import pdb
import sklearn

def get_detection_dataset(X_test_all, Y_test, X_test_adv_list, failed_adv_as_positve=True, predict_func=None):
    X_test_adv_all = np.vstack(X_test_adv_list)
    # Try to get a balanced dataset.
    X_test_leg = X_test_all[:min(len(X_test_adv_all), len(X_test_all))]
    X_detect = np.vstack([X_test_leg, X_test_adv_all])
    print ("Detection dataset: %d legitimate examples, %d adversarial examples." % (len(X_test_leg), len(X_test_adv_all)))

    if failed_adv_as_positve:
        Y_detect = np.hstack([np.zeros(len(X_test_leg)), np.ones(len(X_test_adv_all))])
    else:
        # Label the correctly recognized adversarial examples as legitimate.
        y_test_adv_all_label = np.array(list(np.argmax(Y_test, axis=1)) * len(X_test_adv_list))
        y_test_adv_all_pred = np.argmax(predict_func(X_test_adv_all), axis=1)
        failed_adv_idx = np.where(y_test_adv_all_label == y_test_adv_all_pred)

        Y_test_adv_all_corrected = np.ones(len(X_test_adv_all))
        Y_test_adv_all_corrected[failed_adv_idx] = 0

        Y_detect = np.hstack([np.zeros(len(X_test_leg)), Y_test_adv_all_corrected])

        print ("Labeled %d failed adversarial examples as %s" % (len(failed_adv_idx[0]) ,'positive' if failed_adv_as_positve else 'negative'))

    return X_detect, Y_detect


# TODO: Add X_train if we have more adversarial examples.
def get_balanced_detection_dataset(X_test_all, Y_test, X_test_adv_list, predict_func=None):
    X_test_adv_all = np.vstack(X_test_adv_list)
    # Try to get a balanced dataset.
    X_test_leg = X_test_all[:min(len(X_test_adv_all), len(X_test_all))]
    X_detect = np.vstack([X_test_leg, X_test_adv_all])
    print ("Detection dataset: %d legitimate examples, %d adversarial examples." % (len(X_test_leg), len(X_test_adv_all)))

    Y_detect = np.hstack([np.zeros(len(X_test_leg)), np.ones(len(X_test_adv_all))])

    # Identify the correctly recognized adversarial examples.
    y_test_adv_all_label = np.array(list(np.argmax(Y_test, axis=1)) * len(X_test_adv_list))
    y_test_adv_all_pred = np.argmax(predict_func(X_test_adv_all), axis=1)
    failed_adv_idx = np.where(y_test_adv_all_label == y_test_adv_all_pred)

    failed_adv_idx = failed_adv_idx[0]
    if len(failed_adv_idx) > 0:
        failed_adv_idx += len(X_test_leg)

    # Y_detect_neg_identified = Y_detect.copy()
    # Y_detect_neg_identified[failed_adv_idx] = 0

    return X_detect, Y_detect, failed_adv_idx


def get_train_test_idx(train_ratio, length):
    random.seed(1234)
    train_idx = random.sample(range(length), int(train_ratio*length))
    test_idx = [ i for i in range(length) if i not in train_idx]
    return train_idx, test_idx


def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    return TP/np.sum(true_labels), FP/np.sum(1-true_labels)


def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = sklearn.metrics.accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr

