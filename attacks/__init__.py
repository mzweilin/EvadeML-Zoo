from future.standard_library import install_aliases
install_aliases()
from urllib import parse as urlparse

import pickle
import numpy as np
import os
import time

from .cleverhans_wrapper import generate_fgsm_examples, generate_jsma_examples, generate_bim_examples
from .carlini_wrapper import generate_carlini_l2_examples, generate_carlini_li_examples, generate_carlini_l0_examples
from .deepfool_wrapper import generate_deepfool_examples, generate_universal_perturbation_examples
from .adaptive.adaptive_adversary import generate_adaptive_carlini_l2_examples

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

def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]


# TODO: replace pickle with .h5 for Python 2/3 compatibility issue.
def maybe_generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, use_cache=False, verbose=True, attack_log_fpath=None):
    x_adv_fpath = use_cache
    if use_cache and os.path.isfile(x_adv_fpath):
        print ("Loading adversarial examples from [%s]." % os.path.basename(x_adv_fpath))
        X_adv, duration = pickle.load(open(x_adv_fpath, "rb"))
    else:
        time_start = time.time()
        X_adv = generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath)
        duration = time.time() - time_start

        if not isinstance(X_adv, np.ndarray):
            X_adv, aux_info = X_adv
        else:
            aux_info = {}

        aux_info['duration'] = duration

        if use_cache:
            pickle.dump((X_adv, aux_info), open(x_adv_fpath, 'wb'))
    return X_adv, duration


def parse_attack_string(attack_string):
    if '?' in attack_string:
        attack_name, attack_params = attack_string.split('?')
    else:
        attack_name, attack_params = attack_string, ''
    attack_name = attack_name.lower()
    attack_params = urlparse.parse_qs(attack_params)
    attack_params = dict( (k, v if len(v)>1 else v[0] ) for k,v in attack_params.items())

    for k,v in attack_params.items():
        if k in ['batch_size', 'max_iterations', 'num_classes', 'max_iter', 'nb_iter', 'max_iter_df']:
            attack_params[k] = int(v)
        elif v == 'true':
            attack_params[k] = True
        elif v == 'false':
            attack_params[k] = False
        elif v == 'inf':
            attack_params[k] = np.inf
        elif isfloat(v):
            attack_params[k] = float(v)
    return attack_name, attack_params

def generate_adv_examples(sess, model, x, y, X, Y, attack_name, attack_params, verbose, attack_log_fpath):
    if attack_name == 'fgsm':
        generate_adv_examples_func = generate_fgsm_examples
    elif attack_name == 'jsma':
        generate_adv_examples_func = generate_jsma_examples
    elif attack_name == 'bim':
        generate_adv_examples_func = generate_bim_examples
    elif attack_name == 'carlinil2':
        generate_adv_examples_func = generate_carlini_l2_examples
    elif attack_name == 'carlinili':
        generate_adv_examples_func = generate_carlini_li_examples
    elif attack_name == 'carlinil0':
        generate_adv_examples_func = generate_carlini_l0_examples
    elif attack_name == 'deepfool':
        generate_adv_examples_func = generate_deepfool_examples
    elif attack_name == 'unipert':
        generate_adv_examples_func = generate_universal_perturbation_examples
    elif attack_name == 'adaptive_carlini_l2':
        generate_adv_examples_func = generate_adaptive_carlini_l2_examples
    else:
        raise NotImplementedError("Unsuported attack [%s]." % attack_name)

    X_adv = generate_adv_examples_func(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath)

    return X_adv

