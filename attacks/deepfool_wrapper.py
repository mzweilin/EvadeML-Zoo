import tensorflow as tf
import numpy as np
from keras.models import Model

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_externals
from deepfool import deepfool

import warnings
import click

def override_params(default, update):
    for key in default:
        if key in update:
            val = update[key]
            default[key] = val
            del update[key]

    if len(update) > 0:
        warnings.warn("Ignored arguments: %s" % update.keys())
    return default

def generate_deepfool_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    """
    Untargeted attack. Y is not needed.
    """
    nb_classes = Y.shape[1]

    f = model.predict

    model_logits = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    persisted_input = x
    persisted_output = model_logits(x)

    print('>> Compiling the gradient tensorflow functions. This might take some time...')
    scalar_out = [tf.slice(persisted_output, [0, i], [1, 1]) for i in range(0, nb_classes)]
    dydx = [tf.gradients(scalar_out[i], [persisted_input])[0] for i in range(0, nb_classes)]

    print('>> Computing gradient function...')
    def grad_fs(image_inp, inds): return [sess.run(dydx[i], feed_dict={persisted_input: image_inp}) for i in inds]

    params = {'num_classes': 10, 'overshoot': 0.02, 'max_iter': 50}
    params = override_params(params, attack_params)

    adv_x_list = []
    aux_info = {}
    aux_info['r_tot'] = []
    aux_info['loop_i'] = []
    aux_info['k_i'] = []

    with click.progressbar(range(0, len(X)), file=sys.stderr, show_pos=True,
                           width=40, bar_template='  [%(bar)s] DeepFool Attacking %(info)s',
                           fill_char='>', empty_char='-') as bar:
        # Loop over the samples we want to perturb into adversarial examples
        for i in bar:
            image = X[i:i+1,:,:,:]
            r_tot, loop_i, k_i, pert_image = deepfool(image, f, grad_fs, **params)
            adv_x_list.append(pert_image)

            aux_info['r_tot'].append(r_tot)
            aux_info['loop_i'].append(loop_i)
            aux_info['k_i'].append(k_i)

    return np.vstack(adv_x_list), aux_info