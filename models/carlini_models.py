from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import MaxPooling2D, Conv2D


def carlini_mnist_model(logits=True, scaling = True):
    input_shape=(28, 28, 1)
    nb_filters = 32
    nb_denses = [200,200,10]
    return carlini_model(input_shape, nb_filters, nb_denses, logits=logits, scaling = scaling)


def carlini_cifar10_model(logits=True, scaling = True):
    input_shape=(32, 32, 3)
    nb_filters = 64
    nb_denses = [256,256,10]
    return carlini_model(input_shape, nb_filters, nb_denses, logits=logits, scaling = scaling)


def carlini_model(input_shape, nb_filters, nb_denses, logits=True, scaling = True):
    """
    :params logits: no softmax layer if True.
    :params scaling: expect [-0.5,0.5] input range if True, otherwise [0, 1]
    """

    model = Sequential()

    if not scaling:
        model.add(Lambda(lambda x: x-0.5, input_shape=input_shape))
    else:
        model.add(Lambda(lambda x: x, input_shape=input_shape))

    model.add(Conv2D(nb_filters, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(nb_filters*2, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters*2, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(nb_denses[0]))
    model.add(Activation('relu'))
    model.add(Dense(nb_denses[1]))
    model.add(Activation('relu'))
    model.add(Dense(nb_denses[2]))

    if not logits:
        model.add(Activation('softmax'))

    return model