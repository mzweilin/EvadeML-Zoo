import sys, os
import click
import pdb
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals

from utils.output import disablePrint, enablePrint


class CarliniModelWrapper:
    def __init__(self, logits, image_size, num_channels, num_labels):
        """
        :image_size: (e.g., 28 for MNIST, 32 for CIFAR)
        :num_channels: 1 for greyscale, 3 for color images
        :num_labels: total number of valid labels (e.g., 10 for MNIST/CIFAR)
        """
        self.logits = logits
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_labels =  num_labels

        # self.model = model_mnist_logits(img_rows=image_size, img_cols=image_size, nb_filters=64, nb_classes=num_labels)
        self.model = logits

    def predict(self, X):
        """
        Run the prediction network *without softmax*.
        """
        return self.model(X)

from keras.models import Model
from keras.layers import Lambda, Input

def convert_model(model, input_shape):
    # Output model: accept [-0.5, 0.5] input range instead of [0,1], output logits instead of softmax.
    # The output model will have three layers in abstract: Input, Lambda, TrainingModel.
    model_logits = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    input_tensor = Input(shape=input_shape)

    scaler = lambda x: x+0.5
    scaler_layer = Lambda(scaler, input_shape=input_shape)(input_tensor)
    output_tensor = model_logits(scaler_layer)

    model_new = Model(inputs=input_tensor, outputs=output_tensor)
    return model_new


def wrap_to_carlini_model(model, X, Y):
    image_size, num_channels = X.shape[1], X.shape[3]
    num_labels = Y.shape[1]
    model_logits = convert_model(model, input_shape=X.shape[1:])
    model_wrapper = CarliniModelWrapper(model_logits, image_size=image_size, num_channels=num_channels, num_labels=num_labels)
    return model_wrapper


from nn_robust_attacks.l2_attack import CarliniL2
def generate_carlini_l2_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    model_wrapper = wrap_to_carlini_model(model, X, Y)

    accepted_params = ['batch_size', 'confidence', 'targeted', 'learning_rate', 'binary_search_steps', 'max_iterations', 'abort_early', 'initial_const']
    for k in attack_params:
        if k not in accepted_params:
            raise NotImplementedError("Unsuporrted params in Carlini L2: %s" % k)

    # assert batch_size <= len(X)
    if 'batch_size' in attack_params and attack_params['batch_size'] > len(X):
        attack_params['batch_size'] = len(X)

    if 'binary_search_steps' in attack_params:
        attack_params['binary_search_steps'] = int(attack_params['binary_search_steps'])

    attack = CarliniL2(sess, model_wrapper, **attack_params)

    if not verbose:
        disablePrint(attack_log_fpath)
    # The input range is [0, 1], convert to [-0.5, 0.5] by subtracting 0.5.
    # The return range is [-0.5, 0.5]. Convert back to [0,1] by adding 0.5.
    X_adv = attack.attack(X - 0.5, Y) + 0.5
    if not verbose:
        enablePrint()

    return X_adv


from nn_robust_attacks.li_attack import CarliniLi
def generate_carlini_li_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    model_wrapper = wrap_to_carlini_model(model, X, Y)

    if 'batch_size' in attack_params:
        batch_size = attack_params['batch_size']
        del attack_params['batch_size']
    else:
        batch_size= 10

    accepted_params = ['targeted', 'learning_rate', 'max_iterations', 'abort_early', 'initial_const', 'largest_const', 'reduce_const', 'decrease_factor', 'const_factor', 'confidence']
    for k in attack_params:
        if k not in accepted_params:
            raise NotImplementedError("Unsuporrted params in Carlini Li: %s" % k)

    attack = CarliniLi(sess, model_wrapper, **attack_params)
    
    X_adv_list = []

    with click.progressbar(range(0, len(X)), file=sys.stderr, show_pos=True, 
                           width=40, bar_template='  [%(bar)s] Carlini Li Attacking %(info)s', 
                           fill_char='>', empty_char='-') as bar:
        for i in bar:
            if i % batch_size == 0:
                X_sub = X[i:min(i+batch_size, len(X)),:]
                Y_sub = Y[i:min(i+batch_size, len(X)),:]
                if not verbose:
                    disablePrint(attack_log_fpath)
                X_adv_sub = attack.attack(X_sub - 0.5, Y_sub) + 0.5
                if not verbose:
                    enablePrint()
                X_adv_list.append(X_adv_sub)

    X_adv = np.vstack(X_adv_list)
    return X_adv


from nn_robust_attacks.l0_attack import CarliniL0
def generate_carlini_l0_examples(sess, model, x, y, X, Y, attack_params, verbose, attack_log_fpath):
    model_wrapper = wrap_to_carlini_model(model, X, Y)

    if 'batch_size' in attack_params:
        batch_size = attack_params['batch_size']
        del attack_params['batch_size']
    else:
        batch_size= 10

    accepted_params = ['targeted', 'learning_rate', 'max_iterations', 'abort_early', 'initial_const', 'largest_const', 'reduce_const', 'decrease_factor', 'const_factor', 'independent_channels', 'confidence']
    for k in attack_params:
        if k not in accepted_params:
            raise NotImplementedError("Unsuporrted params in Carlini L0: %s" % k)

    attack = CarliniL0(sess, model_wrapper, **attack_params)

    X_adv_list = []

    with click.progressbar(range(0, len(X)), file=sys.stderr, show_pos=True, 
                           width=40, bar_template='  [%(bar)s] Carlini L0 Attacking %(info)s', 
                           fill_char='>', empty_char='-') as bar:
        for i in bar:
            if i % batch_size == 0:
                X_sub = X[i:min(i+batch_size, len(X)),:]
                Y_sub = Y[i:min(i+batch_size, len(X)),:]
                if not verbose:
                    disablePrint(attack_log_fpath)
                X_adv_sub = attack.attack(X_sub - 0.5, Y_sub) + 0.5
                if not verbose:
                    enablePrint()
                X_adv_list.append(X_adv_sub)

    X_adv = np.vstack(X_adv_list)
    return X_adv