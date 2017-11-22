"""
The model is adapted from
  https://github.com/MadryLab/mnist_challenge/blob/master/model.py
"""

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Reshape, Dense, Activation

def pgdtrained_mnist_model(logits=True, input_range_type=1, pre_filter=lambda x:x):
    model = Sequential()

    input_shape = (28, 28, 1)
  
    if input_range_type == 1:
        # The input data range is [0, 1]. 
        # Don't need to do scaling for cleverhans models, as it is the input range by default.
        scaler = lambda x: x
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5]. Convert to [0,1] by adding 0.5 element-wise.
        scaler = lambda x: x+0.5
    elif input_range_type == 3:
        # The input data range is [-1, 1]. Convert to [0,1] by x/2+0.5.
        scaler = lambda x: x/2+0.5

    model.add(Lambda(scaler, input_shape=input_shape))
    model.add(Lambda(pre_filter, output_shape=input_shape))

    model.add(Conv2D(filters=32, 
                   kernel_size=(5, 5),
                   strides=(1, 1),
                   padding='same',
                   activation='relu',
                  ))
    model.add(MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                        ))

    model.add(Conv2D(filters=64, 
                   kernel_size=(5, 5),
                   strides=(1, 1),
                   padding='same',
                   activation='relu',
                  ))
    model.add(MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                        ))

    model.add(Reshape((7*7*64,)))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=10))
    if not logits:
        model.add(Activation('softmax'))

    return model
