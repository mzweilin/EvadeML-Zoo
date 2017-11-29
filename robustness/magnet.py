from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

from tensorflow.python.platform import flags
FLAGS = flags.FLAGS

from utils.parameter_parser import parse_params
from externals.MagNet.worker import SimpleReformer

mnist_autoencoder_fpath = os.path.join(project_path, "downloads/MagNet/defensive_models/MNIST_I")
cifar10_autoencoder_fpath = os.path.join(project_path, "downloads/MagNet/defensive_models/CIFAR")

class MagNetRC:
    def __init__(self, keras_model, rc_name):
        # Example of rc_name: FeatureSqueezing?squeezer=bit_depth_1
        self.model_predict = lambda x: keras_model.predict(x)
        subject, params = parse_params(rc_name)
        assert subject == 'MagNet'

        if FLAGS.dataset_name == "MNIST":
            self.reformer = SimpleReformer(mnist_autoencoder_fpath)
        elif FLAGS.dataset_name == "CIFAR-10":
            self.reformer = SimpleReformer(cifar10_autoencoder_fpath)

        self.filter = lambda x: self.reformer.heal(x)

    def predict(self, X):
        X_filtered = self.filter(X)
        Y_pred = self.model_predict(X_filtered)
        return Y_pred

    def visualize_and_predict(self, X):
        X_filtered = self.filter(X)
        Y_pred = self.model_predict(X_filtered)
        return X_filtered, Y_pred