
import warnings
from .pgd_attack import LinfPGDAttack

from keras.models import Model
import tensorflow as tf
import numpy as np

def override_params(default, update):
    for key in default:
        if key in update:
            val = update[key]
            if key == 'k':
                val = int(val)
            default[key] = val
            del update[key]

    if len(update) > 0:
        warnings.warn("Ignored arguments: %s" % update.keys())
    return default


class PGDModelWrapper:
    def __init__(self, keras_model, x, y):
        model_logits = Model(inputs=keras_model.layers[0].input, outputs=keras_model.layers[-2].output)

        self.x_input = x
        self.y_input = tf.argmax(y, 1)
        self.pre_softmax = model_logits(x)

        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)
        self.xent = tf.reduce_sum(y_xent)

        self.y_pred = tf.argmax(self.pre_softmax, 1)


def generate_pgdli_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    model_for_pgd = PGDModelWrapper(model, x, y)
    params = {'model': model_for_pgd, 'epsilon': 0.3, 'k':40, 'a':0.01, 'random_start':True,
                     'loss_func':'xent' }
    params = override_params(params, attack_params)
    attack = LinfPGDAttack(**params)

    Y_class = np.argmax(Y, 1)
    X_adv = attack.perturb(X, Y_class, sess)
    return X_adv
