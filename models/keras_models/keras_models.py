import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

def scaling_tf(X, input_range_type):
    """
    Convert to [0, 255], then subtracting means, convert to BGR.
    """

    if input_range_type == 1:
        # The input data range is [0, 1]. 
        # Convert to [0, 255] by multiplying 255
        X = X*255
    elif input_range_type == 2:
        # The input data range is [-0.5, 0.5]. Convert to [0,255] by adding 0.5 element-wise.
        X = (X+0.5)*255
    elif input_range_type == 3:
        # The input data range is [-1, 1]. Convert to [0,1] by x/2+0.5.
        X = (X/2+0.5)*255

    # Caution: Resulting in zero gradients.
    # X_uint8 = tf.clip_by_value(tf.rint(X), 0, 255)
    red, green, blue = tf.split(X, 3, 3)
    X_bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
            # TODO: swap 0 and 2. should be 2,1,0 according to Keras' original code.
        ], 3)

    # x[:, :, :, 0] -= 103.939
    # x[:, :, :, 1] -= 116.779
    # x[:, :, :, 2] -= 123.68
    return X_bgr

# It looks Keras-Lambda layer doesn't support numpy operations.
from keras.applications.imagenet_utils import preprocess_input
def scaling_np(X, scaling=False):
    if scaling:
        X = X + 0.5
    X_uint8 = np.clip(np.rint(X*255), 0, 255)
    X_bgr = preprocess_input(X_uint8)
    return X_bgr

from .resnet50_model import ResNet50
def keras_resnet50_imagenet_model(logits=False, input_range_type=1):
    """
    Run the prediction network *without softmax*.
    """
    input_shape = (224, 224, 3)
    # if scaling:
    #     x = x + 0.5
    # x_bgr = scaling_tf(x)
    model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None, classes=1000, logits=logits, input_range_type=input_range_type)
    # predictions = model.outputs[0]
    # return predictions
    return model

from .vgg19_model import VGG19
def keras_vgg19_imagenet_model(logits=False, input_range_type=1):
    """
    Run the prediction network *without softmax*.
    """
    input_shape = (224, 224, 3)
    model = VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling=None, classes=1000, logits=logits, input_range_type=input_range_type)
    return model

from .inceptionv3_model import InceptionV3
def keras_inceptionv3_imagenet_model(logits=False, input_range_type=1):
    input_shape = (299, 299, 3)
    model = InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=input_shape,
                pooling=None,
                classes=1000,
                logits=logits,
                input_range_type=input_range_type)
    return model

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    model = keras_resnet50_imagenet_model(x)