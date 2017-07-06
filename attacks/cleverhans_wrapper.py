import numpy as np
import tensorflow as tf
import click

import pdb
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals
from cleverhans.utils_tf import model_loss, batch_eval

import warnings

def override_params(default, update):
    for key in default:
        if key in update:
            val = update[key]
            if key == 'ord':
                if val == 'li':
                    val = np.inf
                elif val == 'l2':
                    val = 2
                elif val == 'l1':
                    val = 1
                else:
                    raise ValueError("Unsuporrted ord: %s" % val)
            default[key] = val
            del update[key]

    if len(update) > 0:
        warnings.warn("Ignored arguments: %s" % update.keys())
    return default


from cleverhans.attacks import FastGradientMethod
def generate_fgsm_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    """
    Untargeted attack. Y is not needed.
    """
    fgsm = FastGradientMethod(model, back='tf', sess=sess)
    fgsm_params = {'eps': 0.1, 'ord': np.inf, 'y': None, 'clip_min': 0, 'clip_max': 1}
    fgsm_params = override_params(fgsm_params, attack_params)

    X_adv = fgsm.generate_np(X, **fgsm_params)
    return X_adv


from cleverhans.attacks import BasicIterativeMethod
def generate_bim_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    """
    Untargeted attack. Y is not needed.
    """
    bim = BasicIterativeMethod(model, back='tf', sess=sess)
    bim_params = {'eps': 0.1, 'eps_iter':0.05, 'nb_iter':10, 'y':y,
                     'ord':np.inf, 'clip_min':0, 'clip_max':1 }
    bim_params = override_params(bim_params, attack_params)

    X_adv = bim.generate_np(X, **bim_params)
    return X_adv


from cleverhans.attacks import SaliencyMapMethod
def generate_jsma_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    """
    Targeted attack, with target classes in Y.
    """
    Y_target = Y

    nb_classes = Y.shape[1]

    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'nb_classes': nb_classes, 'clip_min': 0.,
                   'clip_max': 1., 'targets': y,
                   'y_val': None}
    jsma_params = override_params(jsma_params, attack_params)

    adv_x_list = []

    with click.progressbar(range(0, len(X)), file=sys.stderr, show_pos=True, 
                           width=40, bar_template='  [%(bar)s] JSMA Attacking %(info)s', 
                           fill_char='>', empty_char='-') as bar:
        # Loop over the samples we want to perturb into adversarial examples
        for sample_ind in bar:
            sample = X[sample_ind:(sample_ind+1)]

            jsma_params['y_val'] = Y_target[[sample_ind],]
            adv_x = jsma.generate_np(sample, **jsma_params)
            adv_x_list.append(adv_x)

    return np.vstack(adv_x_list)

