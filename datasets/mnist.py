import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_externals
from cleverhans.utils_mnist import data_mnist

from models.carlini_models import carlini_mnist_model
from models.cleverhans_models import cleverhans_mnist_model


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
        """
        :params logits: no softmax layer if True.
        :params scaling: expect [-0.5,0.5] input range if True, otherwise [0, 1]
        """
        if model_name not in ["cleverhans", 'cleverhans_adv_trained', 'carlini']:
            raise ("Undefined model [%s] for %s." % (model_name, self.dataset_name))
        self.model_name = model_name

        model_weights_fpath = "%s_%s.keras_weights.h5" % (self.dataset_name, model_name)
        model_weights_fpath = os.path.join('trained_models', model_weights_fpath)

        # self.maybe_download_model()
        if model_name in ["cleverhans", 'cleverhans_adv_trained']:
            model = cleverhans_mnist_model(logits=logits, scaling=scaling)
        else:
            model = carlini_mnist_model(logits=logits, scaling = scaling)
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
