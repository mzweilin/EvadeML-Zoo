import pickle
import urlparse
import numpy as np
import os

from cleverhans_wrapper import generate_fgsm_examples, generate_jsma_examples, generate_bim_examples

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]


def maybe_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, use_cache = False):
    if use_cache:
        x_adv_fpath = use_cache

        if os.path.isfile(x_adv_fpath):
            X_adv = pickle.load(open(x_adv_fpath))
        else:
            X_adv = generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params)
            pickle.dump(X_adv, open(x_adv_fpath, 'wb'))
    else:
        X_adv = generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params)
    return X_adv

def parse_attack_string(attack_string):
    if '?' in attack_string:
        attack_name, attack_params = attack_string.split('?')
    else:
        attack_name, attack_params = attack_string, ''
    attack_name = attack_name.lower()
    attack_params = urlparse.parse_qs(attack_params)
    attack_params = dict( (k, v if len(v)>1 else v[0] ) for k,v in attack_params.iteritems())

    for k,v in attack_params.iteritems():
        if isfloat(v):
            attack_params[k] = float(v)
    return attack_name, attack_params

def generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params):
    batch_size = 100
    if attack_name == 'fgsm':
        X_adv = generate_fgsm_examples(sess, model, x, X, batch_size, attack_params)
    elif attack_name == 'jsma':
        X_adv = generate_jsma_examples(sess, model, x, y, X, Y, attack_params)
    elif attack_name == 'bim':
        X_adv = generate_bim_examples(sess, model, x, y, X, Y, attack_params)
    elif attack_name == 'carlini_l2':
        X_adv = generate_Carlini_l2_examples(sess, predictions, x, X, attack_params)

    return X_adv

