from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.squeeze import get_squeezer_by_name

from .base import parse_params

class FeatureSqueezingRC:
    def __init__(self, keras_model, rc_name):
        # Example of rc_name: FeatureSqueezing?squeezer=bit_depth_1
        self.model_predict = lambda x: keras_model.predict(x)
        subject, params = parse_params(rc_name)
        assert subject == 'FeatureSqueezing'
        self.filter = get_squeezer_by_name(params['squeezer'], 'python')        

    def predict(self, X):
        X_filtered = self.filter(X)
        Y_pred = self.model_predict(X_filtered)
        return Y_pred