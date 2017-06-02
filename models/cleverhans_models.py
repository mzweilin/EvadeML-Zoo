import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals
from cleverhans.utils import conv_2d

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import MaxPooling2D, Conv2D

def cleverhans_mnist_model(logits=True, scaling = True):
    input_shape = (28, 28, 1)
    nb_filters = 64
    nb_classes = 10
    return cleverhans_model(input_shape, nb_filters, nb_classes, logits=logits, scaling=scaling)

def cleverhans_cifar10_model(logits=True, scaling = True):
    input_shape = (32, 32, 3)
    nb_filters = 64
    nb_classes = 10
    return cleverhans_model(input_shape, nb_filters, nb_classes, logits=logits, scaling=scaling)

def cleverhans_model(input_shape, nb_filters, nb_classes, logits=False, scaling=False):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param scaling: input [0,1] if False, else [-0.5, 0.5]
    :return:
    """
    model = Sequential()

    if scaling is True:
        # Convert from float [-0.5,0.5] to float [0,1]
        layers = [Lambda(lambda x:x+0.5, input_shape=input_shape), Dropout(0.2)]
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