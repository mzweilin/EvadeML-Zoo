from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pdb
import pickle
import time

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags



FLAGS = flags.FLAGS

# Arguments for task scheduling
flags.DEFINE_string('dataset_name', 'MNIST', 'dataset name.')
# flags.DEFINE_string('model_name', 'cleverhans', '')
flags.DEFINE_string('model_name', 'carlini', '')
flags.DEFINE_string('attacks', "FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next&batch_size=10&max_iterations=1000;CarliniL2?targeted=next&batch_size=10&max_iterations=1000&confidence=2", '')
# flags.DEFINE_string('attacks', "CarliniL2?targeted=next&batch_size=100&max_iterations=1000", '')
flags.DEFINE_boolean('visualize', True, 'Output the image examples as image or not.')
flags.DEFINE_string('defense', 'feature_squeezing', '')
flags.DEFINE_integer('nb_examples', 100, '')
flags.DEFINE_string('result_folder', "./results", '')
flags.DEFINE_boolean('test_mode', True, '')
# flags.DEFINE_string('', '', '')



def load_tf_session():
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)
    print("Created TensorFlow session and set Keras backend.")
    return sess

def main(argv=None):
    # 1. Load a dataset.
    from datasets import MNISTDataset, CIFAR10Dataset
    from datasets import get_correct_prediction_idx, calculate_mean_confidence, calculate_accuracy, calculate_mean_distance

    if FLAGS.dataset_name == "MNIST":
        dataset = MNISTDataset()
    elif FLAGS.dataset_name == "CIFAR-10":
        dataset = CIFAR10Dataset()
    elif FLAGS.dataset_name == "ImageNet":
        dataset = ImageNetDataset()

    print ("\n===Loading %s data..." % FLAGS.dataset_name)
    X_test_all, Y_test_all = dataset.get_test_dataset()


    # 2. Load a trained model.
    sess = load_tf_session()
    keras.backend.set_learning_phase(0)
    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, dataset.image_size, dataset.image_size, dataset.num_channels))
    y = tf.placeholder(tf.float32, shape=(None, dataset.num_classes))

    with tf.variable_scope(FLAGS.model_name):
        model = dataset.load_model_by_name(FLAGS.model_name, logits=False, scaling=False)
        model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])

        model_carlini = dataset.load_model_by_name(FLAGS.model_name, logits=True, scaling=True)
        model_carlini.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])


    # 3. Evaluate the trained model.
    Y_pred_all = model.predict(X_test_all)
    mean_conf_all = calculate_mean_confidence(Y_pred_all, Y_test_all)
    # _, accuracy_all = model.evaluate(X_test_all, Y_test_all, batch_size=128)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    print('Mean confidence on ground truth classes %.4f' % (mean_conf_all))


    # 4. Select some examples to attack.
    # TODO: select the target class: least likely, next, all?
    from datasets import get_first_example_id_each_class
    # It doesn't make sense to attack with a misclassified example.
    correct_idx = get_correct_prediction_idx(Y_pred_all, Y_test_all)
    if FLAGS.test_mode:
        # Only select the first example of each class.
        correct_and_selected_idx = get_first_example_id_each_class(Y_test_all[correct_idx])
        selected_idx = [ correct_idx[i] for i in correct_and_selected_idx ]
    else:
        selected_idx = correct_idx[:FLAGS.nb_examples]

    X_test, Y_test, Y_pred = X_test_all[selected_idx], Y_test_all[selected_idx], Y_pred_all[selected_idx]

    accuracy_selected = calculate_accuracy(Y_pred, Y_test)
    mean_conf_selected = calculate_mean_confidence(Y_pred, Y_test)
    print('Test accuracy on selected legitimate examples %.4f' % (accuracy_selected))
    print('Mean confidence on ground truth classes, selected %.4f' % (mean_conf_selected))
    


    # 5. Generate adversarial examples.
    from attacks import maybe_generate_adv_examples, parse_attack_string

    # Generate i + 1 (mod 10) as the target classes.
    from attacks import get_next_class
    Y_test_target_next = get_next_class(Y_test)

    X_test_adv_list = []

    attack_string_list = filter(lambda x:len(x)>0, FLAGS.attacks.split(';'))
    for attack_string in attack_string_list:
        attack_name, attack_params = parse_attack_string(attack_string)

        if 'targeted' in attack_params:
            targeted = attack_params['targeted']
            Y_test_target = Y_test_target_next
        else:
            targeted = False
            Y_test_target = Y_test

        if 'carlini' in attack_name:
            target_model = model_carlini
        else:
            target_model = model

        x_adv_fname = "%s_%d_%s_%s.pickle" % (FLAGS.dataset_name, len(X_test), FLAGS.model_name, attack_string)
        x_adv_fpath = os.path.join(FLAGS.result_folder, x_adv_fname)
        # x_adv_fpath = False
            

        time_start = time.time()
        X_test_adv = maybe_generate_adv_examples(sess, target_model, x, y, X_test, Y_test_target, attack_name, attack_params, use_cache = x_adv_fpath)
        duration = time.time() - time_start
        X_test_adv_list.append(X_test_adv)

        # 5.1. Evaluate the quality of adversarial examples
        Y_test_adv_pred = model.predict(X_test_adv)


        # pdb.set_trace()
        success_rate = calculate_accuracy(Y_test_adv_pred, Y_test_target)
        mean_conf = calculate_mean_confidence(Y_test_adv_pred, Y_test_target)
        if targeted is False:
            success_rate = 1 - success_rate
            mean_conf = 1 - mean_conf

        mean_l2_dist, mean_li_dist, mean_l0_dist = calculate_mean_distance(X_test, X_test_adv)

        dur_per_sample = duration / len(X_test_adv)

        print ("\n---Attack: %s" % attack_string)
        print ("Success rate: %.2f%%, Mean confidence: %.2f%%" % (success_rate*100, mean_conf*100))
        num_pixels = X_test.shape[1] * X_test.shape[2]
        l0_percent = float(mean_l0_dist/num_pixels) * 100
        print ("L2 dist: %.4f, Li dist: %.4f, L0 dist: %d/%d (%.1f%%)" % (mean_l2_dist, mean_li_dist, mean_l0_dist, num_pixels, l0_percent))
        print ("Duration: %.4f per sample" % dur_per_sample)

    if FLAGS.visualize is True:
        from datasets.visualization import show_imgs_in_rows
        selected_idx = get_first_example_id_each_class(Y_test)
        
        legitimate_examples = X_test[selected_idx]
        rows = [legitimate_examples]
        rows += map(lambda x:x[selected_idx], X_test_adv_list)

        img_fpath = os.path.join(FLAGS.result_folder, '%s_%s_adv_examples.png' % (dataset.dataset_name, dataset.model_name) )
        show_imgs_in_rows(rows, img_fpath)
        print ('\n===Adversarial image examples are saved in ', img_fpath)


    # 6. Evaluate defense techniques.
    if FLAGS.defense == 'feature_squeezing':
        """
        Test the accuracy with feature squeezing filters.
        """
        from defenses.feature_squeezing.robustness import calculate_squeezed_accuracy
        
        for attack_string, X_test_adv in zip(FLAGS.attacks, X_test_adv_list):
            csv_fpath = "%s_%d_%s_%s_robustness.csv" % (dataset.dataset_name, FLAGS.nb_examples, dataset.model_name, attack_string)
            csv_fpath = os.path.join(FLAGS.result_folder, csv_fpath)

            print ("\n===Calculating the accuracy with feature squeezing...")
            calculate_squeezed_accuracy(model, Y_test, X_test, X_test_adv, csv_fpath)
            print ("\n---Results are stored in ", csv_fpath, '\n')


        """
        Test the detection.
        """
        from defenses.feature_squeezing.detection import train_a_adversarial_detector

        X_test_adv_all = np.vstack(X_test_adv_list)
        # Try to get a balanced dataset.
        X_test_leg = X_test_all[:min(len(X_test_adv_all), len(X_test_all))]

        X_for_detection = np.vstack([X_test_leg, X_test_adv_all])
        Y_for_detection = np.hstack([np.zeros(len(X_test_leg)), np.ones(len(X_test_adv_all))])
        print ("Detection dataset: %d legitimate examples, %d adversarial examples" % (len(X_test_leg), len(X_test_adv_all)))
        train_a_adversarial_detector(model, X_for_detection, Y_for_detection)

        print ("---Done")

if __name__ == '__main__':
    main()
