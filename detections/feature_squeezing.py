import sklearn
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy.stats import entropy
from keras.models import Model

import operator
import functools
import pdb
import random
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from utils.visualization import draw_plot
from utils.squeeze import get_squeezer_by_name
from utils.parameter_parser import parse_params

def reshape_2d(x):
    if len(x.shape) > 2:
        # Reshape to [#num_examples, ?]
        batch_size = x.shape[0]
        num_dim = functools.reduce(operator.mul, x.shape, 1)
        x = x.reshape((batch_size, num_dim/batch_size))
    return x


# Normalization.
# Two approaches: 1. softmax; 2. unit-length vector (unit norm).

# Code Source: ?
def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


from sklearn.preprocessing import normalize
def unit_norm(x):
    """
    x: a 2D array: (batch_size, vector_length)
    """
    return normalize(x, axis=1)


l1_dist = lambda x1,x2: np.sum(np.abs(x1 - x2), axis=tuple(range(len(x1.shape))[1:]))
l2_dist = lambda x1,x2: np.sum((x1-x2)**2, axis=tuple(range(len(x1.shape))[1:]))**.5


# Note: KL-divergence is not symentric.
# Designed for probability distribution (e.g. softmax output).
def kl(x1, x2):
    assert x1.shape == x2.shape
    # x1_2d, x2_2d = reshape_2d(x1), reshape_2d(x2)

    # Transpose to [?, #num_examples]
    x1_2d_t = x1.transpose()
    x2_2d_t = x2.transpose()

    # pdb.set_trace()
    e = entropy(x1_2d_t, x2_2d_t)
    e[np.where(e==np.inf)] = 2
    return e


class FeatureSqueezingDetector:
    def __init__(self, model, param_str):
        self.model = model
        subject, params = parse_params(param_str)

        layer_id = len(model.layers)-1
        normalizer = 'none'
        metric = params['distance_measure']
        squeezers_name = params['squeezers'].split(',')
        self.set_config(layer_id, normalizer, metric, squeezers_name)

        if params.has_key('threshold'):
            self.threshold = float(params['threshold'])
        else:
            self.threshold = None
            self.train_fpr = float(params['fpr'])

    def get_squeezer_by_name(self, name):
        return get_squeezer_by_name(name, 'python')

    def get_normalizer_by_name(self, name):
        d = {'unit_norm': unit_norm, 'softmax': softmax, 'none': lambda x:x}
        return d[name]

    def get_metric_by_name(self, name):
        d = {'kl_f': lambda x1,x2: kl(x1, x2), 'kl_b': lambda x1,x2: kl(x2, x1), 'l1': l1_dist, 'l2': l2_dist}
        return d[name]

    def set_config(self, layer_id, normalizer_name, metric_name, squeezers_name):
        self.layer_id = layer_id
        self.normalizer_name = normalizer_name
        self.metric_name = metric_name
        self.squeezers_name = squeezers_name

    def get_config(self):
        return self.layer_id, self.normalizer_name, self.metric_name, self.squeezers_name

    # Visualize the propagation of perturbations.
    # Scenerio 1: Assume we have a perfect squeezer that always recover adversarial example to legitimate. The distance of legitimate is zero.
    # Scenerio 2: Use one(or several) feature squeezer(s) that barely affect the legitimate example. The distance of legitimate may be positive.
    def view_adv_propagation(self, X, X_adv, squeezers_name):
        """
        Assume we have a perfeect feature squeezer that always recover a given adversariale example to the legitimate version.
        The distance of legitimate is zero then.
        We want to find out which layer has the most different output between the adversarial and legitimate example pairs, 
            under several measurements.
        """
        model = self.model

        for layer in model.layers:
            shape_size = functools.reduce(operator.mul, layer.output_shape[1:])
            print (layer.name, shape_size)

        xs = np.arange(len(model.layers))

        ret = []

        for normalizer in ['unit_norm', 'softmax', 'none']:
            normalize_func = self.get_normalizer_by_name(normalizer)
            label_list = []
            series_list = []

            if normalizer == "softmax":
                metric_list = ['kl_f', 'kl_b', 'l1', 'l2']
            else:
                metric_list = ['l1', 'l2']
            for distance_metric_name in metric_list:
                distance_func = self.get_metric_by_name(distance_metric_name)

                series = []

                for layer_id in range(len(model.layers)):
                    self.set_config(layer_id, normalizer, distance_metric_name, squeezers_name)

                    if len(squeezers_name) > 0:
                        # With feature squeezers: Scenerio 2.
                        distance = self.get_distance(X_adv) - self.get_distance(X)
                    else:
                        # Assume a perfect feature squeezer: Scenerio 1.
                        distance = self.get_distance(X, X_adv)
                    mean_dist = np.mean(distance)
                    series.append(mean_dist)

                series = np.array(series).astype(np.double)
                series = series/np.max(series)
                series_list.append(series)
                label_list.append("%s_%s" % (normalizer, distance_metric_name))

                layer_id = np.argmax(series)
                print ("Best: Metric-%s at Layer-%d, normalized by %s" % (distance_metric_name, layer_id, normalizer))
                ret.append([layer_id, normalizer, distance_metric_name])

            draw_plot(xs, series_list, label_list, "./%s_%s.png" % (self.name_prefix, normalizer))

        return ret

    def calculate_distance_max(self, val_orig, vals_squeezed, metric_name):
        distance_func = self.get_metric_by_name(metric_name)

        dist_array = []
        for val_squeezed in vals_squeezed:
            dist = distance_func(val_orig, val_squeezed)
            dist_array.append(dist)

        dist_array = np.array(dist_array)
        return np.max(dist_array, axis=0)

    def get_distance(self, X1, X2=None):
        layer_id, normalizer_name, metric_name, squeezers_name = self.get_config()

        normalize_func = self.get_normalizer_by_name(normalizer_name)
        input_to_normalized_output = lambda x: normalize_func(reshape_2d(self.eval_layer_output(x, layer_id)))

        val_orig_norm = input_to_normalized_output(X1)

        if X2 is None:
            vals_squeezed = []
            for squeezer_name in squeezers_name:
                squeeze_func = self.get_squeezer_by_name(squeezer_name)
                val_squeezed_norm = input_to_normalized_output(squeeze_func(X1))
                vals_squeezed.append(val_squeezed_norm)
            distance = self.calculate_distance_max(val_orig_norm, vals_squeezed, metric_name)
        else:
            val_1_norm = val_orig_norm
            val_2_norm = input_to_normalized_output(X2)
            distance_func = self.get_metric_by_name(metric_name)
            distance = distance_func(val_1_norm, val_2_norm)

        return distance

    def eval_layer_output(self, X, layer_id):
        layer_output = Model(inputs=self.model.layers[0].input, outputs=self.model.layers[layer_id].output)
        return layer_output.predict(X)


    def output_distance_csv(self, X_list, field_name_list, csv_fpath):
        from utils.output import write_to_csv
        distances_list = []
        for X in X_list:
            distances = self.get_distance(X)
            distances_list.append(distances)

        to_csv = []
        for i in range(len(X_list[0])):
            record = {}
            for j, field_name in enumerate(field_name_list):
                if len(distances_list[j]) > i:
                    record[field_name] = distances_list[j][i]
                else:
                    record[field_name] = None
            to_csv.append(record)

        write_to_csv(to_csv, csv_fpath, field_name_list)


    # Only examine the legitimate examples to get the threshold, ensure low False Positive rate.
    def train(self, X, Y):
        """
        Calculating distance depends on:
            layer_id
            normalizer
            distance metric
            feature squeezer(s)
        """

        if self.threshold is not None:
            print ("Loaded a pre-defined threshold value %f" % self.threshold)
        else:
            layer_id, normalizer_name, metric_name, squeezers_name = self.get_config()

            neg_idx = np.where(Y == 0)[0]
            X_neg = X[neg_idx]
            distances = self.get_distance(X_neg)

            selected_distance_idx = int(np.ceil(len(X_neg) * (1-self.train_fpr)))
            threshold = sorted(distances)[selected_distance_idx-1]
            self.threshold = threshold
            print ("Selected %f as the threshold value." % self.threshold)
        return self.threshold

    def test(self, X):
        layer_id, normalizer_name, metric_name, squeezers_name = self.get_config()

        distances = self.get_distance(X)
        threshold = self.threshold
        Y_pred = distances > threshold

        return Y_pred, distances
