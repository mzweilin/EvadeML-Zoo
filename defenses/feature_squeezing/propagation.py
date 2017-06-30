from keras.models import Model
import numpy as np
from .detection import reshape_2d, softmax, unit_norm, kl, l1_dist, l2_dist

import matplotlib.pyplot as plt
# import pdb

import functools, operator

def draw_plot(xs, series_list, label_list, fname):
    fig, ax = plt.subplots()

    for i,series in enumerate(series_list):
        smask = np.isfinite(series)
        ax.plot(xs[smask], series[smask], linestyle='-', marker='o', label=label_list[i])

    legend = ax.legend(loc='upper right', shadow=True)

    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # plt.show()
    plt.savefig(fname)
    plt.close(fig)

    # return fig


def view_propagation(X, X_adv, model, name, squeezers):
    def eval_layer_output(X, layer_id):
        layer_output = Model(inputs=model.layers[0].input, outputs=model.layers[layer_id].output)
        return layer_output.predict(X)

    xs = np.arange(len(model.layers))
    series_list = []
    # label_list = []

    # pdb.set_trace()
    for layer in model.layers:
        shape_size = functools.reduce(operator.mul, layer.output_shape[1:])
        print (layer.name, shape_size)



    # for normalizer in ['unit_norm', 'none']:
    # Fact: Multi-round softmax cancel the difference. 
    for normalizer in ['unit_norm', 'softmax', 'none']:
        # for distance_metric_name in ['kl_f', 'kl_b', 'l1', 'l2']:
        label_list = []
        series_list = []
        for distance_metric_name in ['l1', 'l2', ]:
            if normalizer == 'unit_norm':
                normalize_func = unit_norm
            elif normalizer == 'softmax':
                normalize_func = softmax
            else:
                normalize_func = lambda x:x

            if distance_metric_name == 'l1':
                distance_func = l1_dist
            elif distance_metric_name == 'l2':
                distance_func = l2_dist
            elif distance_metric_name == 'kl_f':
                distance_func = lambda x1,x2: kl(x1, x2)
            elif distance_metric_name == 'kl_b':
                distance_func = lambda x1,x2: kl(x2, x1)

            series = []

            for layer_id in range(len(model.layers)):
                z = eval_layer_output(X, layer_id)
                z_adv = eval_layer_output(X_adv, layer_id)
                z_squeezed = eval_layer_output(X_squeezed, layer_id)

                z, z_adv, z_squeezed = normalize_func(reshape_2d(z)), normalize_func(reshape_2d(z_adv)), normalize_func(reshape_2d(z_squeezed))

                # l1_distance = l1_dist(z, z_adv)
                l2_distance = distance_func(z, z_adv) - distance_func(z, z_squeezed)

                # , np.std(l2_distance)
                # print ("%d,%f" % (layer_id, np.mean(l2_distance)))
                # print (np.mean(l2_distance))
                mean_l2_dist = np.mean(l2_distance)
                series.append(mean_l2_dist)


            series = np.array(series).astype(np.double)
            # if np.min(series) < 0:
            #     series += np.min(series)
            series = series/np.max(series)
            series_list.append(series)
            label_list.append("%s_%s" % (normalizer, distance_metric_name))


        draw_plot(xs, series_list, label_list, "./%s_%s.png" % (name, normalizer))

        # fig.savefig("./%s.png" % normalizer)
        # plt.close(fig)





# xs = np.arange(8)
# series1 = np.array([1, 3, 3, None, None, 5, 8, 9]).astype(np.double)
# s1mask = np.isfinite(series1)
# series2 = np.array([2, None, 5, None, 4, None, 3, 2]).astype(np.double)
# s2mask = np.isfinite(series2)

# plt.plot(xs[s1mask], series1[s1mask], linestyle='-', marker='o')
# plt.plot(xs[s2mask], series2[s2mask], linestyle='-', marker='o')

# plt.show()