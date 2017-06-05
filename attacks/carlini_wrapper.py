import sys, os
import pdb
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals
from nn_robust_attacks import l2_attack#, li_attack, l0_attack

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


from nn_robust_attacks.l2_attack import CarliniL2
def generate_carlini_l2_examples(sess, model_logits, x, y, X, Y, attack_params):
    image_size, num_channels = X.shape[1], X.shape[3]
    num_labels = Y.shape[1]

    model_wrapper = CarliniModelWrapper(model_logits, image_size=image_size, num_channels=num_channels, num_labels=num_labels)

    accepted_params = ['batch_size', 'confidence', 'targeted', 'learning_rate', 'binary_search_steps', 'max_iterations', 'abort_early', 'initial_const']
    for k in attack_params:
        if k not in accepted_params:
            raise ("Unsuporrted params in Carlini L2: %s" % k)

    # assert batch_size <= len(X)
    if 'batch_size' in attack_params and attack_params['batch_size'] > len(X):
        attack_params['batch_size'] = len(X)

    attack = CarliniL2(sess, model_wrapper, **attack_params)
    X_scaled = X - 0.5
    X_adv = attack.attack(X_scaled, Y)
    return X_adv + 0.5


from nn_robust_attacks.li_attack import CarliniLi
def generate_carlini_li_examples(sess, model_logits, x, y, X, Y, attack_params):
    image_size, num_channels = X.shape[1], X.shape[3]
    num_labels = Y.shape[1]

    model_wrapper = CarliniModelWrapper(model_logits, image_size=image_size, num_channels=num_channels, num_labels=num_labels)

    if 'batch_size' in attack_params:
        batch_size = attack_params['batch_size']
        del attack_params['batch_size']
    else:
        batch_size= 10

    accepted_params = ['targeted', 'learning_rate', 'max_iterations', 'abort_early', 'initial_const', 'largest_const', 'reduce_const', 'decrease_factor', 'const_factor']
    for k in attack_params:
        if k not in accepted_params:
            raise ("Unsuporrted params in Carlini Li: %s" % k)

    attack = CarliniLi(sess, model_wrapper, **attack_params)
    
    X_adv_list = []
    for i in np.arange(0, len(X), batch_size):
        X_sub = X[i:min(i+batch_size, len(X)),:]
        X_scaled = X_sub - 0.5
        X_adv_sub = attack.attack(X_scaled, Y)
        X_adv_list.append(X_adv_sub)

    X_adv = np.vstack(X_adv_list)
    return X_adv + 0.5


from nn_robust_attacks.l0_attack import CarliniL0
def generate_carlini_l0_examples(sess, model_logits, x, y, X, Y, attack_params):
    image_size, num_channels = X.shape[1], X.shape[3]
    num_labels = Y.shape[1]

    model_wrapper = CarliniModelWrapper(model_logits, image_size=image_size, num_channels=num_channels, num_labels=num_labels)

    if 'batch_size' in attack_params:
        batch_size = attack_params['batch_size']
        del attack_params['batch_size']
    else:
        batch_size= 10

    accepted_params = ['targeted', 'learning_rate', 'max_iterations', 'abort_early', 'initial_const', 'largest_const', 'reduce_const', 'decrease_factor', 'const_factor', 'independent_channels']
    for k in attack_params:
        if k not in accepted_params:
            raise ("Unsuporrted params in Carlini L0: %s" % k)

    attack = CarliniL0(sess, model_wrapper, **attack_params)

    X_adv_list = []
    for i in np.arange(0, len(X), batch_size):
        X_sub = X[i:min(i+batch_size, len(X)), :]
        X_scaled = X_sub - 0.5
        X_adv_sub = attack.attack(X_scaled, Y)
        X_adv_list.append(X_adv_sub)

    X_adv = np.vstack(X_adv_list)
    return X_adv + 0.5