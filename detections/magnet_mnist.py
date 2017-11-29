## test_defense.py -- test defense
##
## Copyright (C) 2017, Dongyu Meng <zbshfmmm@gmail.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Load external module: MagNet
import sys, os
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from externals.MagNet.setup_mnist import MNIST
from externals.MagNet.utils import prepare_data
from externals.MagNet.worker import AEDetector, SimpleReformer, IdReformer, AttackData, Classifier, Operator, Evaluator

import numpy as np
import os

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.activations import softmax

class ClassifierWrapper:
    def __init__(self, model):
        """
        Keras classifier wrapper.
        Note that the wrapped classifier should spit logits as output.
        """
        layer_id = len(model.layers)-2
        self.model = Model(inputs=model.layers[0].input, outputs=model.layers[layer_id].output)
        self.softmax = Sequential()
        self.softmax.add(Lambda(lambda X: softmax(X, axis=1), input_shape=(10,)))

    def classify(self, X, option="logit", T=1):
        if option == "logit":
            return self.model.predict(X)
        if option == "prob":
            logits = self.model.predict(X)/T
            return self.softmax.predict(logits)

    def print(self):
        return "Classifier:"+self.path.split("/")[-1]

class MagNetDetector:
    def __init__(self, model, detector_name):
        cur_folder = os.path.dirname(os.path.abspath(__file__))

        autoencoder_model_I_fpath = os.path.join(project_dir, "downloads/MagNet/defensive_models/MNIST_I")
        autoencoder_model_II_fpath = os.path.join(project_dir, "downloads/MagNet/defensive_models/MNIST_II")

        detector_I = AEDetector(autoencoder_model_I_fpath, p=2)
        detector_II = AEDetector(autoencoder_model_II_fpath, p=1)
        reformer = SimpleReformer(autoencoder_model_I_fpath)

        id_reformer = IdReformer()
        classifier = ClassifierWrapper(model)

        detector_dict = dict()
        detector_dict["I"] = detector_I
        detector_dict["II"] = detector_II

        self.operator = Operator(MNIST(), classifier, detector_dict, reformer)

    def train(self, X=None, Y=None, fpr=None):
        drop_rate={"I": 0.001, "II": 0.001}
        # drop_rate={"I": fpr*0.5, "II": fpr*0.5}
        print("\n==========================================================")
        print("Drop Rate:", drop_rate)
        self.thrs = self.operator.get_thrs(drop_rate)
        print("Thresholds:", self.thrs)

    def test(self, X):
        all_pass, detector_breakdown = self.operator.filter(X, self.thrs)
        print ("detector_breakdown", detector_breakdown)
        ret_detection = np.array([ False if i in all_pass else True for i in range(len(X)) ])
        return ret_detection, ret_detection

if __name__ == '__main__':
    magnet_detector = MagNetDetector()
    magnet_detector.train()

    X = magnet_detector.operator.data.test_data
    Y_detected, _ = magnet_detector.test(X)

    print ("False positive rate: %f" % (np.sum(Y_detected)/float(len(X))))
