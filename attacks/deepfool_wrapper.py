import tensorflow as tf
import numpy as np
from keras.models import Model

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_externals
from deepfool import deepfool

def generate_deepfool_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    """
    Untargeted attack. Y is not needed.
    """
    nb_classes = Y.shape[1]

    f = model.predict

    model_logits = Model(inputs=model.input, outputs=model.layers[-2].output)

    persisted_input = x
    persisted_output = model_logits(x)

    print('>> Compiling the gradient tensorflow functions. This might take some time...')
    scalar_out = [tf.slice(persisted_output, [0, i], [1, 1]) for i in range(0, nb_classes)]
    dydx = [tf.gradients(scalar_out[i], [persisted_input])[0] for i in range(0, nb_classes)]

    print('>> Computing gradient function...')
    def grad_fs(image_inp, inds): return [sess.run(dydx[i], feed_dict={persisted_input: image_inp}) for i in inds]

    adv_x_list = []
    for i in range(len(X)):
        image = X[i:i+1,:,:,:]
        r_tot, loop_i, k_i, pert_image = deepfool(image, f, grad_fs, num_classes=10, overshoot=0.02, max_iter=50)
        adv_x_list.append(pert_image)

    return np.vstack(adv_x_list)