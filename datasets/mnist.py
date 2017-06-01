import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals
from cleverhans.utils import conv_2d
from cleverhans.utils_mnist import data_mnist


# From Cleverhans.
def scaler(x):
    """
    Convert from float [-0.5,0.5] to float [0,1]
    """
    return x + 0.5


def scaler_output_shape(input_shape):
    return input_shape


def cnn_model(logits=False, img_rows=28, img_cols=28, channels=1, nb_filters=64, nb_classes=10, scaling=False):
    """
    [A modified model from Cleverhans]
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    if scaling is True:
        layers = [Lambda(scaler, input_shape=input_shape), Dropout(0.2)]
    else:
        layers = [Dropout(0.2, input_shape=input_shape)]

    layers += [conv_2d(nb_filters, (8, 8), (2, 2), "same"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
              Activation('relu'),
              conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
              Activation('relu'),
              Dropout(0.5),
              Flatten(),
              Dense(nb_classes)]

    for layer in layers:
        model.add(layer)

    if not logits:
        model.add(Activation('softmax'))

    return model


class MNISTDataset:
    def __init__(self):
        self.dataset_name = "MNIST"
        self.image_size = 28
        self.num_channels = 1
        self.num_classes = 10            
        # self.maybe_download_model = maybe_download_mnist_model
        self.result_folder = 'results/mnist'

    def get_test_dataset(self):
        _, _, X_test, Y_test = data_mnist()
        X_test = X_test.reshape(X_test.shape[0], self.image_size, self.image_size, self.num_channels)
        return X_test, Y_test

    def load_model_by_name(self, model_name, logits=False, scaling=False):
        if model_name not in ["cleverhans", 'cleverhans_adv_trained']:
            raise ("Undefined model [%s] for %s." % (model_name, self.dataset_name))
        self.model_name = model_name

        model_weights_fpath = "%s_%s.keras_weights.h5" % (self.dataset_name, model_name)
        model_weights_fpath = os.path.join('trained_models', model_weights_fpath)

        # self.maybe_download_model()
        model = cnn_model(logits=logits, img_rows=28, img_cols=28,
                channels=1, nb_filters=64, nb_classes=10, scaling=scaling)
        print("\n===Defined TensorFlow model graph.")
        model.load_weights(model_weights_fpath)
        print ("---Loaded MNIST-%s model.\n" % model_name)
        return model

if __name__ == '__main__':
    # from datasets.mnist import *
    dataset = MNISTDataset()
    X_test, Y_test = dataset.get_test_dataset()
    print (X_test.shape)
    print (Y_test.shape)

    model_name = 'cleverhans'
    model = dataset.load_model_by_name(model_name)

    model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
    _,accuracy = model.evaluate(X_test, Y_test, batch_size=128)
    print ("\nTesting accuracy: %.4f" % accuracy)