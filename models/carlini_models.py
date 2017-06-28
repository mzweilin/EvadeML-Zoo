from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import MaxPooling2D, Conv2D


def carlini_mnist_model(logits=True, input_range_type=2, pre_filter=lambda x:x):
    input_shape=(28, 28, 1)
    nb_filters = 32
    nb_denses = [200,200,10]
    return carlini_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter)


def carlini_cifar10_model(logits=True, input_range_type=2, pre_filter=lambda x:x):
    input_shape=(32, 32, 3)
    nb_filters = 64
    nb_denses = [256,256,10]
    return carlini_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter)


def carlini_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter):
    """
    :params logits: return logits(input of softmax layer) if True; return softmax output otherwise.
    :params input_range_type: the expected input range, {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
    """

    model = Sequential()

    if input_range_type == 1:
        # The input data range is [0, 1]. 
        # Convert to [-0.5,0.5] by x-0.5.
        scaler = lambda x: x-0.5
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5].
        # Don't need to do scaling for carlini models, as it is the input range by default.
        scaler = lambda x: x
    elif input_range_type == 3:
        # The input data range is [-1, 1]. Convert to [-0.5,0.5] by x/2.
        scaler = lambda x: x/2

    model.add(Lambda(scaler, input_shape=input_shape))
    model.add(Lambda(pre_filter, output_shape=input_shape))

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