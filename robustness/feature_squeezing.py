from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.squeeze import get_squeezer_by_name
from utils.parameter_parser import parse_params

class FeatureSqueezingRC:
    def __init__(self, keras_model, rc_name):
        # Example of rc_name: FeatureSqueezing?squeezer=bit_depth_1
        self.model_predict = lambda x: keras_model.predict(x)
        subject, params = parse_params(rc_name)
        assert subject == 'FeatureSqueezing'

        if params.has_key('squeezer'):
            self.filter = get_squeezer_by_name(params['squeezer'], 'python')
        elif params.has_key('squeezers'):
            squeezer_names = params['squeezers'].split(',')
            self.filters = [ get_squeezer_by_name(squeezer_name, 'python') for squeezer_name in squeezer_names ]

            def filter_func(x, funcs):
                x_p = x
                for func in funcs:
                    x_p = func(x_p)
                return x_p

            self.filter = lambda x: filter_func(x, self.filters)



    def predict(self, X):
        X_filtered = self.filter(X)
        Y_pred = self.model_predict(X_filtered)
        return Y_pred

    def visualize_and_predict(self, X):
        X_filtered = self.filter(X)
        Y_pred = self.model_predict(X_filtered)
        return X_filtered, Y_pred