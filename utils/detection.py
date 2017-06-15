import numpy as np
import random

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

def get_train_test_idx(train_ratio, length):
    random.seed(1234)
    train_idx = random.sample(range(length), int(train_ratio*length))
    test_idx = [ i for i in range(length) if i not in train_idx]
    return train_idx, test_idx