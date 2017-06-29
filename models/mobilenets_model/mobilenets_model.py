import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import load_externals

from mobilenets import *
from mobilenets import _obtain_input_shape, __conv_block, __depthwise_conv_block

from keras.layers import Lambda

def scaling_tf(X, input_range_type):
    """
    Convert to [-1, 1].
    """
    if input_range_type == 1:
        # The input data range is [0, 1]. Convert to [-1, 1] by
        X = X - 0.5
        X = X * 2.
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5]. Convert to [-1,1] by
        X = X * 2.
    elif input_range_type == 3:
        # The input data range is [-1, 1].
        X = X

    return X


def __create_mobilenet(classes, img_input, include_top, alpha, depth_multiplier, dropout, pooling, logits):
    ''' Creates a MobileNet model with specified parameters
    Args:
        classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        alpha: width multiplier of the MobileNet.
        depth_multiplier: depth multiplier for depthwise convolution
                          (also called the resolution multiplier)
        dropout: dropout rate
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''

    x = __conv_block(img_input, 32, alpha, strides=(2, 2))
    x = __depthwise_conv_block(x, 64, alpha, depth_multiplier, id=1)

    x = __depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), id=2)
    x = __depthwise_conv_block(x, 128, alpha, depth_multiplier, id=3)

    x = __depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), id=4)
    x = __depthwise_conv_block(x, 256, alpha, depth_multiplier, id=5)

    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), id=6)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=7)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=8)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=9)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=10)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=11)

    x = __depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), id=12)
    x = __depthwise_conv_block(x, 1024, alpha, depth_multiplier, id=13)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Convolution2D(classes, (1, 1), padding='same', name='conv_preds')(x)
        # Reshape from (?, 1, 1, 1000)  to (?, 1000)
        x = Reshape((classes,), name='reshape_2')(x)

        # Move Reshape before Actionvation. Otherwise, Cleverhans gets confused in fetching logits output.
        if not logits:
            x = Activation('softmax', name='activation')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    return x


def MobileNets(input_shape=None, alpha=1.0, depth_multiplier=1,
               dropout=1e-3, include_top=True, weights='imagenet',
               input_tensor=None, pooling=None, classes=1000,
               logits=False, input_range_type=1, pre_filter=lambda x:x):
    ''' Instantiate the MobileNet architecture.
        Note that only TensorFlow is supported for now,
        therefore it only works with the data format
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or (3, 224, 224) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(200, 200, 3)` would be one valid value.
            alpha: width multiplier of the MobileNet.
            depth_multiplier: depth multiplier for depthwise convolution
                (also called the resolution multiplier)
            dropout: dropout rate
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or
                `imagenet` (ImageNet weights)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        '''

    if K.backend() == 'theano':
        raise AttributeError('Theano backend is not currently supported, '
                             'as Theano does not support depthwise convolution yet.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top`'
                         ' as true, `classes` should be 1001')

    if weights == 'imagenet':
        assert depth_multiplier == 1, "If imagenet weights are being loaded, depth multiplier must be 1"

        assert alpha in [0.25, 0.50, 0.75, 1.0], "If imagenet weights are being loaded, alpha can be one of" \
                                                 "`0.25`, `0.50`, `0.75` or `1.0` only."

        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        rows, cols = (0, 1) if K.image_data_format() == 'channels_last' else (1, 2)

        rows = int(input_shape[rows])
        cols = int(input_shape[cols])

        assert rows == cols and rows in [None, 128, 160, 192, 224], "If imagenet weights are being loaded," \
                                                                    "image must be square and be one of " \
                                                                    "(128,128), (160,160), (192,192), or (224, 224)." \
                                                                    "Given (%d, %d)" % (rows, cols)

    # Determine proper input shape. Note, include_top is False by default, as
    # input shape can be anything larger than 32x32 and the same number of parameters
    # will be used.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      include_top=False)

    # If input shape is still None, provide a default input shape
    if input_shape is None:
        input_shape = (224, 224, 3) if K.image_data_format() == 'channels_last' else (3, 224, 224)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Scaling
    # x = __create_mobilenet(classes, img_input, include_top, alpha, depth_multiplier, dropout, pooling, logits)
    x = Lambda(lambda x: scaling_tf(x, input_range_type))(img_input)
    x = Lambda(pre_filter, output_shape=input_shape)(x)
    x = __create_mobilenet(classes, x, include_top, alpha, depth_multiplier, dropout, pooling, logits)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='mobilenet')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise AttributeError('Weights for Channels Last format are not available')

        if (alpha == 1.) and (depth_multiplier == 1.):
            if include_top:
                model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
                weigh_path = BASE_WEIGHT_PATH + model_name
                weights_path = get_file(model_name,
                                        weigh_path,
                                        cache_subdir='models')
            else:
                model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
                weigh_path = BASE_WEIGHT_PATH + model_name
                weights_path = get_file(model_name,
                                        weigh_path,
                                        cache_subdir='models')

            model.load_weights(weights_path)

    return model


def mobilenet_imagenet_model(logits=False, input_range_type=1, pre_filter=None):
    input_shape = (224, 224, 3)
    model = MobileNets(input_shape=input_shape, alpha=1.0, depth_multiplier=1,
               dropout=1e-3, include_top=True, weights='imagenet',
               input_tensor=None, pooling=None, classes=1000,
               logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
    return model