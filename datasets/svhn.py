import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('datasets/svhn_dataset/')
import download_svhn_data

import numpy as np
import scipy.io as sio

from keras.utils import np_utils

from models.carlini_models import carlini_mnist_model
from models.cleverhans_models import cleverhans_mnist_model
from models.pgdtrained_models import pgdtrained_mnist_model
from models.tohinz_models import tohinz_svhn_model
	
class SVHNDataset:
	def __init__(self):
		self.dataset_name = "SVHN"
		self.image_size = 32
		self.num_channels = 3
		self.num_classes = 10
		
	def get_test_dataset(self):
		test_path = "test_32x32.mat"
                test_path = os.path.join('datasets/svhn_dataset', test_path)
		test = sio.loadmat(test_path)
		X_test = test['X']
		y_test = test['y']
		y_test[y_test == 10] = 0

                y_test = y_test.ravel()

		X_test = np.transpose(X_test,(3,0,1,2))
		X_test = X_test.astype('float32') / 255
		Y_test = np_utils.to_categorical(y_test)
		
		del y_test
		return X_test, Y_test
		
	def get_val_dataset(self):
		train_path = "train_32x32.mat"
                train_path = os.path.join('datasets/svhn_dataset', train_path)
		train = sio.loadmat(train_path)
		X_train = train['X']
		y_train = train['y']
		y_train[y_train == 10] = 0

		y_train = y_train.ravel()
		val_size = 5000

		
		X_val = X_train[:val_size]
		X_val = np.transpose(X_val,(3,0,1,2))
                X_val = X_val.astype('float32') / 255
                y_val = y_train[:val_size]
                Y_val = np_utils.to_categorical(y_val)
                del X_train, y_train
                return X_val, Y_val
        
	def load_model_by_name(self, model_name, logits=False, input_range_type=1, pre_filter=lambda x:x):

 		if model_name not in ['tohinz']:
			raise NotImplementedError("Undefined model [%s] for %s." % (model_name, self.dataset_name))
                self.model_name = model_name

                model_weights_fpath = "%s_%s.keras_weights.h5" % (self.dataset_name, model_name)
                model_weights_fpath = os.path.join('downloads/trained_models', model_weights_fpath)

                if model_name in ["tohinz"]:
                        model = tohinz_svhn_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
                print("\n===Defined TensorFlow model graph.")
                model.load_weights(model_weights_fpath)
                print ("---Loaded SVHN-%s model.\n" % model_name)
                return model

if __name__ == '__main__':
    # from datasets.mnist import *
    dataset = SVHNDataset()
    X_test, Y_test = dataset.get_test_dataset()
    print (X_test.shape)
    print (Y_test.shape)

    model_name = 'tohinz'
    model = dataset.load_model_by_name(model_name)

    model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
    _,accuracy = model.evaluate(X_test, Y_test, batch_size=128)
    print ("\nTesting accuracy: %.4f" % accuracy)
