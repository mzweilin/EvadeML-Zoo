from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import MaxPooling2D, Conv2D
from keras.layers.normalization import BatchNormalization


def tohinz_svhn_model(logits=False, input_range_type=2, pre_filter=lambda x:x):
    input_shape=(32, 32, 3)
    nb_filters = 32
    nb_denses = [512,10]
    return tohinz_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter)



def tohinz_model(input_shape, nb_filters, nb_denses, logits, input_range_type, pre_filter):
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

    model.add(Conv2D(nb_filters, kernel_size=3, input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(nb_filters*2, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters*2, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(nb_filters*4, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters*4, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(nb_denses[0], activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(nb_denses[1]))	
	
    if not logits:
    	model.add(Activation('softmax'))

    return model
