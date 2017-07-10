from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
import keras.backend as K
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals
from densenet import __transition_block, __dense_block, TF_WEIGHTS_PATH, TF_WEIGHTS_PATH_NO_TOP



def get_densenet_weights_path(dataset_name="CIFAR-10", include_top=True):
    assert dataset_name == "CIFAR-10"
    if include_top:
        weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels.h5',
                                TF_WEIGHTS_PATH,
                                cache_subdir='models')
    else:
        weights_path = get_file('densenet_40_12_tf_dim_ordering_tf_kernels_no_top.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models')
    return weights_path


def densenet_cifar10_model(logits=False, input_range_type=1, pre_filter=lambda x:x):
    assert input_range_type == 1

    batch_size = 64
    nb_classes = 10

    img_rows, img_cols = 32, 32
    img_channels = 3

    img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
    depth = 40
    nb_dense_block = 3
    growth_rate = 12
    nb_filter = 16
    dropout_rate = 0.0 # 0.0 for data augmentation
    input_tensor = None
    include_top=True

    if logits is True:
        activation = None
    else:
        activation = "softmax"

    # Determine proper input shape
    input_shape = _obtain_input_shape(img_dim,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_dense_net(nb_classes, img_input, True, depth, nb_dense_block,
                           growth_rate, nb_filter, -1, False, 0.0,
                           dropout_rate, 1E-4, activation)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')
    return model


# Source: https://github.com/titu1994/DenseNet
def __create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-4,
                       activation='softmax'):
    ''' Build the DenseNet model
    Args:
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4'
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, 'reduction value must lie between 0.0 and 1.0'

    # layers in each dense block
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)  # Convert tuple to list

        assert len(nb_layers) == (nb_dense_block + 1), 'If list, nb_layer is used as provided. ' \
                                                       'Note that list size must be (nb_dense_block + 1)'
        final_nb_layer = nb_layers[-1]
        nb_layers = nb_layers[:-1]
    else:
        if nb_layers_per_block == -1:
            count = int((depth - 4) / 3)
            nb_layers = [count for _ in range(nb_dense_block)]
            final_nb_layer = count
        else:
            final_nb_layer = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * nb_dense_block

    if bottleneck:
        nb_layers = [int(layer // 2) for layer in nb_layers]

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_uniform', padding='same', name='initial_conv2D',
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, bottleneck=bottleneck,
                                     dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = __transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                                 dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(nb_classes, kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
        if activation != None:
            x = Activation(activation)(x)

    return x