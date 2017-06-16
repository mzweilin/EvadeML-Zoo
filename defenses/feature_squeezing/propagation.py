from keras.models import Model
import numpy as np
from .detection import reshape_2d, softmax, unit_norm, kl, l1_dist, l2_dist


def view_propagation(X, X_adv, model):
    def eval_layer_output(X, layer_id):
        layer_output = Model(inputs=model.input, outputs=model.layers[layer_id].output)
        return layer_output.predict(X)

    for layer_id in range(len(model.layers)):
        z = eval_layer_output(X, layer_id)
        z_adv = eval_layer_output(X_adv, layer_id)

        z, z_adv = unit_norm(reshape_2d(z)), unit_norm(reshape_2d(z_adv))

        # l1_distance = l1_dist(z, z_adv)
        l2_distance = l2_dist(z, z_adv)

        # , np.std(l2_distance)
        # print ("%d,%f" % (layer_id, np.mean(l2_distance)))
        print (np.mean(l2_distance))

    print ("")
