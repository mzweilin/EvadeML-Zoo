from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import pdb
import pickle
import time
import random

import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

# Arguments for task scheduling
flags.DEFINE_string('dataset_name', 'MNIST', 'Supported: MNIST, CIFAR-10, ImageNet.')
flags.DEFINE_integer('nb_examples', 100, 'The number of examples selected for attacks.')
flags.DEFINE_boolean('test_mode', False, 'Only select one sample for each class.')
flags.DEFINE_string('model_name', 'carlini', 'Supported: carlini for MNIST and CIFAR-10; cleverhans and cleverhans_adv_trained for MNIST; resnet50 for ImageNet.')
flags.DEFINE_string('attacks', "FGSM?eps=0.1;BIM?eps=0.1&eps_iter=0.02;JSMA?targeted=next;CarliniL2?targeted=next&batch_size=10&max_iterations=1000;CarliniL2?targeted=next&batch_size=10&max_iterations=1000&confidence=2", 'Attack name and parameters in URL style, separated by semicolon.')
flags.DEFINE_boolean('visualize', True, 'Output the image examples for each attack, enabled by default.')
flags.DEFINE_string('defense', 'feature_squeezing1', 'Supported: feature_squeezing.')
flags.DEFINE_string('detection', 'feature_squeezing', 'Supported: feature_squeezing.')
flags.DEFINE_string('result_folder', "results", 'The output folder for results.')
flags.DEFINE_boolean('verbose', False, 'Stdout level. The hidden content will be saved to log files anyway.')


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
    from datasets import MNISTDataset, CIFAR10Dataset, ImageNetDataset
    from datasets import get_correct_prediction_idx, evaluate_adversarial_examples, calculate_mean_confidence, calculate_accuracy

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
        """
        Create two model instances with the same weights. 
          1. "model" for prediction;
          2. "model_carlini" for Carlini/Wagner's attacks.
        The scaling argument, 'input_range_type': {1: [0,1], 2:[-0.5, 0.5], 3:[-1, 1]...}
        """
        model = dataset.load_model_by_name(FLAGS.model_name, logits=False, input_range_type=1)
        model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])

        # Carlini/Wagner's attack implementations require the input range [-0.5, 0.5].
        model_carlini = dataset.load_model_by_name(FLAGS.model_name, logits=True, input_range_type=2)
        model_carlini.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])

    # 3. Evaluate the trained model.
    Y_pred_all = model.predict(X_test_all)
    mean_conf_all = calculate_mean_confidence(Y_pred_all, Y_test_all)
    # _, accuracy_all = model.evaluate(X_test_all, Y_test_all, batch_size=128)
    accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
    print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
    print('Mean confidence on ground truth classes %.4f' % (mean_conf_all))

    # if FLAGS.dataset_name == 'ImageNet':
    #     # TODO: Configure attacks aganist ImageNet models.
    #     return


    # 4. Select some examples to attack.
    import hashlib
    from datasets import get_first_example_id_each_class
    # Filter out the misclassified examples.
    correct_idx = get_correct_prediction_idx(Y_pred_all, Y_test_all)
    if FLAGS.test_mode:
        # Only select the first example of each class.
        correct_and_selected_idx = get_first_example_id_each_class(Y_test_all[correct_idx])
        selected_idx = [ correct_idx[i] for i in correct_and_selected_idx ]
    else:
        selected_idx = correct_idx[:FLAGS.nb_examples]

    from utils.output import format_number_range
    selected_example_idx_ranges = format_number_range(sorted(selected_idx))
    print ( "Selected index in test set (sorted): %s" % selected_example_idx_ranges )

    X_test, Y_test, Y_pred = X_test_all[selected_idx], Y_test_all[selected_idx], Y_pred_all[selected_idx]

    accuracy_selected = calculate_accuracy(Y_pred, Y_test)
    mean_conf_selected = calculate_mean_confidence(Y_pred, Y_test)
    print('Test accuracy on selected legitimate examples %.4f' % (accuracy_selected))
    print('Mean confidence on ground truth classes, selected %.4f\n' % (mean_conf_selected))


    task = {}
    task['dataset_name'] = FLAGS.dataset_name
    task['model_name'] = FLAGS.model_name
    task['accuracy_test'] = accuracy_all
    task['mean_confidence_test'] = mean_conf_all

    task['test_set_selected_length'] = len(selected_idx)
    task['test_set_selected_idx_ranges'] = selected_example_idx_ranges
    task['test_set_selected_idx_hash'] = hashlib.sha1(str(selected_idx).encode('utf-8')).hexdigest()
    task['accuracy_test_selected'] = accuracy_selected
    task['mean_confidence_test_selected'] = mean_conf_selected

    task_id = "%s_%d_%s_%s" % \
            (task['dataset_name'], task['test_set_selected_length'], task['test_set_selected_idx_hash'][:5], task['model_name'])

    FLAGS.result_folder = os.path.join(FLAGS.result_folder, task_id)
    if not os.path.isdir(FLAGS.result_folder):
        os.makedirs(FLAGS.result_folder)

    from utils.output import save_task_descriptor
    save_task_descriptor(FLAGS.result_folder, [task])

    # 5. Generate adversarial examples.
    from attacks import maybe_generate_adv_examples, parse_attack_string
    from defenses.feature_squeezing.squeeze import reduce_precision_np
    attack_string_hash = hashlib.sha1(FLAGS.attacks.encode('utf-8')).hexdigest()[:5]
    sample_string_hash = task['test_set_selected_idx_hash'][:5]

    from attacks import get_next_class, get_least_likely_class
    Y_test_target_next = get_next_class(Y_test)
    Y_test_target_ll = get_least_likely_class(Y_pred)

    X_test_adv_list = []

    attack_string_list = filter(lambda x:len(x)>0, FLAGS.attacks.split(';'))
    to_csv = []

    X_adv_cache_folder = os.path.join(FLAGS.result_folder, 'adv_examples')
    adv_log_folder = os.path.join(FLAGS.result_folder, 'adv_logs')
    for folder in [X_adv_cache_folder, adv_log_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    for attack_string in attack_string_list:
        attack_log_fpath = os.path.join(adv_log_folder, "%s_%s.log" % (task_id, attack_string))
        attack_name, attack_params = parse_attack_string(attack_string)
        print ( "\nRunning attack: %s %s" % (attack_name, attack_params))

        if 'targeted' in attack_params:
            targeted = attack_params['targeted']
            if targeted == 'next':
                Y_test_target = Y_test_target_next
            elif targeted == 'll':
                Y_test_target = Y_test_target_ll
        else:
            targeted = False
            attack_params['targeted'] = False
            # TODO: only adding the param for Carlini's attacks.
            Y_test_target = Y_test

        if 'carlini' in attack_name:
            target_model = model_carlini
        else:
            target_model = model

        x_adv_fname = "%s_%s.pickle" % (task_id, attack_string)
        x_adv_fpath = os.path.join(X_adv_cache_folder, x_adv_fname)

        X_test_adv, duration = maybe_generate_adv_examples(sess, target_model, x, y, X_test, Y_test_target, attack_name, attack_params, use_cache = x_adv_fpath, verbose=FLAGS.verbose, attack_log_fpath=attack_log_fpath)
        X_test_adv_list.append(X_test_adv)

        dur_per_sample = duration / len(X_test_adv)

        # 5.1. Evaluate the quality of adversarial examples
        model_predict = lambda x: model.predict(x)

        print ("\n---Attack: %s" % attack_string)
        rec = evaluate_adversarial_examples(X_test, X_test_adv, Y_test_target.copy(), targeted, model_predict)
        rec['dataset_name'] = FLAGS.dataset_name
        rec['model_name'] = FLAGS.model_name
        rec['attack_string'] = attack_string
        rec['duration_per_sample'] = dur_per_sample
        rec['discretization'] = False
        to_csv.append(rec)

        # 5.2 Adversarial examples being discretized to uint8.
        print ("\n---Attack: %s" % attack_string)
        X_test_adv_discret = reduce_precision_np(X_test_adv, 256)
        rec = evaluate_adversarial_examples(X_test, X_test_adv_discret, Y_test_target.copy(), targeted, model_predict)
        rec['dataset_name'] = FLAGS.dataset_name
        rec['model_name'] = FLAGS.model_name
        rec['attack_string'] = attack_string
        rec['duration_per_sample'] = dur_per_sample
        rec['discretization'] = True
        to_csv.append(rec)


    from utils.output import write_to_csv
    attacks_evaluation_csv_fpath = os.path.join(FLAGS.result_folder, 
            "%s_attacks_%s_evaluation.csv" % \
            (task_id, attack_string_hash))
    fieldnames = ['dataset_name', 'model_name', 'attack_string', 'duration_per_sample', 'discretization', 'success_rate', 'mean_confidence', 'mean_l2_dist', 'mean_li_dist', 'mean_l0_dist_value', 'mean_l0_dist_pixel']
    write_to_csv(to_csv, attacks_evaluation_csv_fpath, fieldnames)

    if FLAGS.visualize is True:
        from datasets.visualization import show_imgs_in_rows
        selected_idx_vis = get_first_example_id_each_class(Y_test)

        legitimate_examples = X_test[selected_idx_vis]
        rows = [legitimate_examples]
        rows += map(lambda x:x[selected_idx_vis], X_test_adv_list)

        img_fpath = os.path.join(FLAGS.result_folder, '%s_attacks_%s_examples.png' % (task_id, attack_string_hash) )
        show_imgs_in_rows(rows, img_fpath)
        print ('\n===Adversarial image examples are saved in ', img_fpath)

    # TODO: 5.5 Investigate how adversarial perturbation propagate throughout layers.

    # Inputs
    # from defenses.feature_squeezing.propagation import view_propagation

    # for X_test_adv in X_test_adv_list:
    #     view_propagation(X_test, X_test_adv, model)

    # view_propagation(X_test, X_test+0.105, model)

    # return

    # 6. Evaluate defense techniques.
    if FLAGS.defense == 'feature_squeezing':
        """
        Test the accuracy with feature squeezing filters.
        """
        from defenses.feature_squeezing.robustness import calculate_squeezed_accuracy

        for attack_string, X_test_adv in zip(attack_string_list, X_test_adv_list):
            csv_fpath = "%s_%s_robustness.csv" % (task_id, attack_string)
            csv_fpath = os.path.join(FLAGS.result_folder, csv_fpath)

            print ("\n===Calculating the accuracy with feature squeezing...")
            calculate_squeezed_accuracy(model, Y_test, X_test, X_test_adv, csv_fpath)
            print ("\n---Results are stored in ", csv_fpath, '\n')


    # 7. Detection experiment. 
    # All data should be discretized to uint8.
    X_test_adv_discretized_list = [ reduce_precision_np(X_test_adv, 256) for X_test_adv in X_test_adv_list]
    del X_test_adv_list

    if FLAGS.detection is not None:
        from utils.detection import get_detection_dataset, get_train_test_idx

        csv_fname = "%s_attacks_%s_detection_%s.csv" % (task_id, attack_string_hash, FLAGS.detection)
        detection_csv_fpath = os.path.join(FLAGS.result_folder, csv_fname)
        to_csv = []
        for failed_adv_as_positve in [True, False]:
            # 7.1 Prepare the dataset for detection.
            X_detect, Y_detect = get_detection_dataset(X_test_all, Y_test, X_test_adv_discretized_list, failed_adv_as_positve, predict_func=model.predict)
            print ("Positive ratio in detection dataset %d/%d" % (np.sum(Y_detect), len(Y_detect)))


            train_ratio = 0.5
            train_idx, test_idx = get_train_test_idx(train_ratio, len(Y_detect))

            X_detect_train, Y_detect_train = X_detect[train_idx], Y_detect[train_idx]
            X_detect_test, Y_detect_test = X_detect[test_idx], Y_detect[test_idx]

            print ("Positive ratio in train %d/%d" % (np.sum(Y_detect_train), len(Y_detect_train)))
            print ("Positive ratio in test %d/%d" % (np.sum(Y_detect_test), len(Y_detect_test)))

            # 7.2 Enumerate all specified detection methods.
            # Feature Squeezing as an example.
            from defenses.feature_squeezing.detection import FeatureSqueezingDetector
            for layer_id in range(len(model.layers)):
                # if layer_id < len(model.layers)-3:
                #     continue
                for normalizer in ['softmax', 'unit_norm', 'none']:
                    for distance_metric in ['kl_f', 'kl_b', 'l1', 'l2']:
                        detector = FeatureSqueezingDetector(model=model, layer_id=layer_id, squeezer_name='median_smoothing_2', distance_metric_name=distance_metric, normalizer=normalizer)
                        detector.train(X_detect_train, Y_detect_train)
                        detect_perf_rec = detector.test(X_detect_test, Y_detect_test)
                        detect_perf_rec['failed_adv_as_positve'] = failed_adv_as_positve
                        detect_perf_rec['layer_id'] = layer_id
                        detect_perf_rec['distance_metric'] = distance_metric
                        detect_perf_rec['normalizer'] = normalizer
                        to_csv.append(detect_perf_rec)

        fieldnames = ['failed_adv_as_positve', 'layer_id', 'distance_metric', 'normalizer', 'roc_auc', 'accuracy', 'tpr', 'fpr', 'threshold']
        write_to_csv(to_csv, detection_csv_fpath, fieldnames)

if __name__ == '__main__':
    main()
