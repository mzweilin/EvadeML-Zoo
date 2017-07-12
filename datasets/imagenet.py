import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import os
# from multiprocessing import Pool
from keras.preprocessing import image

from models.keras_models import keras_resnet50_imagenet_model
from models.keras_models import keras_vgg19_imagenet_model
from models.keras_models import keras_inceptionv3_imagenet_model
from models.mobilenets_model import mobilenet_imagenet_model

# pool = Pool()

def load_single_image(img_path, img_size=224):
    size = (img_size,img_size)
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # Embeded preprocessing in the model.
    # x = preprocess_input(x)
    return x


def _load_single_image(args):
    img_path, img_size = args
    return load_single_image(img_path, img_size)


def data_imagenet(img_folder, img_size, label_style = 'caffe', label_size = 1000, selected_idx = None):
    fnames = os.listdir(img_folder)
    fnames = sorted(fnames, key = lambda x: int(x.split('.')[1]))
    
    if isinstance(selected_idx, list):
        selected_fnames = [fnames[i] for i in selected_idx]
    elif isinstance(selected_idx, int):
        selected_fnames = fnames[:selected_idx]
    else:
        selected_fnames = fnames

    labels = map(lambda x: int(x.split('.')[0]), selected_fnames)
    img_path_list = map(lambda x: [os.path.join(img_folder, x), img_size], selected_fnames)
    X = map(_load_single_image, img_path_list)
    X = np.concatenate(X, axis=0)
    Y = np.eye(1000)[labels]
    return X, Y


class ImageNetDataset:
    def __init__(self):
        self.dataset_name = "ImageNet"
        # self.image_size = 224
        self.num_channels = 3
        self.num_classes = 1000
        self.img_folder = "/tmp/ILSVRC2012_img_val_labeled_caffe"

        if not os.path.isdir:
            raise Exception("Please prepare the ImageNet dataset first: EvadeML-Zoo/datasets/imagenet_dataset/label_as_filename.py.")

    def get_test_dataset(self, img_size=224, num_images=100):
        self.image_size = img_size
        X, Y = data_imagenet(self.img_folder, self.image_size, selected_idx=num_images)
        X /= 255
        return X, Y

    def get_test_data(self, img_size, idx_begin, idx_end):
        # Return part of the dataset.
        self.image_size = img_size
        X, Y = data_imagenet(self.img_folder, self.image_size, selected_idx=range(idx_begin, idx_end))
        X /= 255
        return X, Y

    def load_model_by_name(self, model_name, logits=False, input_range_type=1, input_tensor=None, pre_filter=lambda x:x):
        """
        :params logits: no softmax layer if True.
        :params scaling: expect [-0.5,0.5] input range if True, otherwise [0, 1]
        """
        if model_name == 'resnet50':
            model = keras_resnet50_imagenet_model(logits=logits, input_range_type=input_range_type)
        elif model_name == 'vgg19':
            model = keras_vgg19_imagenet_model(logits=logits, input_range_type=input_range_type)
        elif model_name == 'inceptionv3':
            model = keras_inceptionv3_imagenet_model(logits=logits, input_range_type=input_range_type)
        elif model_name == 'mobilenet':
            model = mobilenet_imagenet_model(logits=logits, input_range_type=input_range_type, pre_filter=pre_filter)
        else:
            raise Exception("Unsupported model: [%s]" % model_name)

        return model

if __name__ == '__main__':
    # label_style = 'caffe'
    # # img_folder = "/mnt/nfs/taichi/imagenet_data/data_val_labeled_%s" % label_style
    # img_folder = "/tmp/ILSVRC2012_img_val_labeled_caffe"
    # X, Y = data_imagenet(img_folder, selected_idx=10)
    # print (X.shape)
    # print (np.argmax(Y, axis=1))

    dataset = ImageNetDataset()

    X, Y = dataset.get_test_dataset()
    model = dataset.load_model_by_name('ResNet50')


    

