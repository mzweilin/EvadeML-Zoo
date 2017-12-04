"""
    Demo if adaptive adversary works against feature squeezing.

    Embed the diffrentiable filter layers in a model.
    Pass in the (average) gradient (part of loss) to an attack algorithm.
    Implement the gaussian-noise-iterative method for non-diffrentiable filter layers (bit depth reduction.)
    Introduce the randomized feature squeezing (need to verify with legitimate examples, should not harm the accuracy.)


"""

import os
import tensorflow as tf
import numpy as np
import math

# Core: Get the gradient of models for the attack algorithms.
#       We will combine the gradient of several models.


from keras.models import Model
from keras.layers import Lambda, Input

def insert_pre_processing_layer_to_model(model, input_shape, func):
    # Output model: accept [-0.5, 0.5] input range instead of [0,1], output logits instead of softmax.
    # The output model will have three layers in abstract: Input, Lambda, TrainingModel.
    model_logits = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    input_tensor = Input(shape=input_shape)

    scaler_layer = Lambda(func, input_shape=input_shape)(input_tensor)
    output_tensor = model_logits(scaler_layer)

    model_new = Model(inputs=input_tensor, outputs=output_tensor)
    return model_new


# maybe_generate_adv_examples(sess, model, x, y, X_test, Y_test_target, attack_name, attack_params, use_cache = x_adv_fpath, verbose=FLAGS.verbose, attack_log_fpath=attack_log_fpath)
def adaptive_attack(sess, model, squeezers, x, y, X_test, Y_test_target, attack_name, attack_params):
    for squeeze_func in squeezers:
        predictions = model(squeeze_func(x))


# tf.contrib.distributions.kl(dist_a, dist_b, allow_nan=False, name=None)

# from .median import median_filter as median_filter_tf
# from .median import median_random_filter as median_random_filter_tf


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.squeeze import get_squeezer_by_name, reduce_precision_tf

# if FLAGS.dataset_name == "MNIST":
#     # squeezers_name = ['median_smoothing_2', 'median_smoothing_3', 'binary_filter']
#     squeezers_name = ['median_smoothing_2', 'binary_filter']
# elif FLAGS.dataset_name == "CIFAR-10":
#     squeezers_name = ["bit_depth_5", "bit_depth_4", 'median_smoothing_1_2', 'median_smoothing_2_1','median_smoothing_2', 'median_smoothing_1_3']
# elif FLAGS.dataset_name == "ImageNet":
#     squeezers_name = ["bit_depth_5", 'median_smoothing_1_2', 'median_smoothing_2_1','median_smoothing_2']

def get_tf_squeezer_by_name(name):
    return get_squeezer_by_name(name, 'tensorflow')

tf_squeezers_name_mnist = ['median_filter_2_2', 'bit_depth_1']
tf_squeezers_name_cifar10 = ['median_filter_1_2', 'median_filter_2_1', 'median_filter_2_2', 'median_filter_1_3', 'bit_depth_5', 'bit_depth_4']
tf_squeezers_name_imagenet = ['median_filter_1_2', 'median_filter_2_1', 'median_filter_2_2', 'median_filter_1_3', 'bit_depth_5']

# tf_squeezers = map(get_tf_squeezer_by_name, tf_squeezers_name)

def get_tf_squeezers_by_str(tf_squeezers_str):
    tf_squeezers_name = tf_squeezers_str.split(',')
    return map(get_tf_squeezer_by_name, tf_squeezers_name)

def kl_tf(x1, x2, eps = 0.000000001):
    x1 = tf.clip_by_value(x1, eps, 1)
    x2 = tf.clip_by_value(x2, eps, 1)
    return tf.reduce_sum(x1 * tf.log(x1/x2), reduction_indices=[1])

def generate_adaptive_carlini_l2_examples(sess, model, x, y, X, Y_target, attack_params, verbose, attack_log_fpath):
    # (model, x, y, X, Y_target, tf_squeezers=tf_squeezers, detector_threshold = 0.2):
    # tf_squeezers=tf_squeezers
    eval_dir = os.path.dirname(attack_log_fpath)

    default_params = {
        'batch_size': 100,
        'confidence': 0,
        'targeted': False,
        'learning_rate': 9e-2,
        'binary_search_steps': 9,
        'max_iterations': 5000,
        'abort_early': False, # TODO: not suported. 
        'initial_const': 0.0,
        'detector_threshold': 0.3,
        'uint8_optimized': False,
        'tf_squeezers': [],
        'distance_measure': 'l1',
        'between_squeezers': False,
    }

    if 'tf_squeezers' in attack_params:
        tf_squeezers_str = attack_params['tf_squeezers']
        tf_squeezers = get_tf_squeezers_by_str(tf_squeezers_str)
        attack_params['tf_squeezers'] = tf_squeezers

    accepted_params = default_params.keys()
    for k in attack_params:
        if k not in accepted_params:
            raise NotImplementedError("Unsuporrted params in Carlini L2: %s" % k)
        else:
            default_params[k] = attack_params[k]

    # assert batch_size <= len(X)
    if 'batch_size' in default_params and default_params['batch_size'] > len(X):
        default_params['batch_size'] = len(X)

    return adaptive_CarliniL2(sess, model, X, Y_target, eval_dir, **default_params)


def adaptive_CarliniL2(sess, model, X, Y_target, eval_dir, batch_size, confidence, targeted, learning_rate, binary_search_steps, max_iterations, abort_early, initial_const, detector_threshold, uint8_optimized, tf_squeezers, distance_measure, between_squeezers):
    model_logits = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    # Need a determined batch size for coefficient vectors.
    x = tf.placeholder(shape=X.shape, dtype=tf.float32)
    y = tf.placeholder(shape=Y_target.shape, dtype=tf.float32)

    # Adapted from Warren and Carlini's code
    N0, H0, W0, C0 = X.shape
    # Range [0, 1], initialize as the original images.
    batch_images = X
    # Get the arctanh of the original images.
    batch_images_tanh = np.arctanh((batch_images - 0.5) / 0.501)
    batch_labels = Y_target

    x_star_tanh = tf.Variable(batch_images_tanh, dtype=tf.float32)
    # Range [0, 1], initialize as the original images.
    x_star = tf.tanh(x_star_tanh) / 2. + 0.5

    # The result is optimized for uint8. 
    x_star_uint8 = reduce_precision_tf(x_star, 256)

    # Gradient required.
    y_pred_logits = model_logits(x_star)
    y_pred = model(x_star)
    print ("tf_squezers: %s" % tf_squeezers)
    y_squeezed_pred_list = [ model(func(x_star)) for func in tf_squeezers ]
    
    coeff = tf.placeholder(shape=(N0,), dtype=tf.float32)
    l2dist = tf.reduce_sum(tf.square(x_star - x), [1, 2, 3])
    ground_truth_logits = tf.reduce_sum(y * y_pred_logits, 1)
    top_other_logits = tf.reduce_max((1 - y) * y_pred_logits - (y * 10000), 1)
    
    # Untargeted attack, minimize the ground_truth_logits.
    # target_penalty = tf.maximum(0., ground_truth_logits - top_other_logits)

    if targeted is False:
        # if untargeted, optimize for making this class least likely.
        target_penalty = tf.maximum(0.0, ground_truth_logits-top_other_logits+confidence)
    else:
        # if targetted, optimize for making the other class most likely
        target_penalty = tf.maximum(0.0, top_other_logits-ground_truth_logits+confidence)
        
        


    # Minimize the sum of L1 score.
    detector_penalty = None

    # TODO: include between squeezers l1.
    all_pred_list = [y_pred] + y_squeezed_pred_list

    if between_squeezers:
        print ("#Between squeezers")
        for i, pred_base in enumerate(all_pred_list):
            for j in range(i+1, len(all_pred_list)):
                pred_target = all_pred_list[j]
                if distance_measure == "l1":
                    score = tf.reduce_sum(tf.abs(pred_base - pred_target), 1)
                elif distance_measure == 'kl_f':
                    score = kl_tf(pred_base, pred_target)
                elif distance_measure == 'kl_b':
                    score = kl_tf(pred_target, pred_base)
                detector_penalty_sub = tf.maximum(0., score - detector_threshold)

                if detector_penalty is None:
                    detector_penalty = detector_penalty_sub
                else:
                    detector_penalty += detector_penalty_sub
    else:
        for y_squeezed_pred in y_squeezed_pred_list:
            if distance_measure == "l1":
                score = tf.reduce_sum(tf.abs(y_pred - y_squeezed_pred), 1)
            elif distance_measure == 'kl_f':
                score = kl_tf(y_pred, y_squeezed_pred)
            elif distance_measure == 'kl_b':
                score = kl_tf(y_squeezed_pred, y_pred)
            detector_penalty_sub = tf.maximum(0., score - detector_threshold)

            if detector_penalty is None:
                detector_penalty = detector_penalty_sub
            else:
                detector_penalty += detector_penalty_sub

    

    # There could be different desion choices. E.g. add one coefficient for the detector penalty.
    loss = tf.add((target_penalty + detector_penalty) * coeff, l2dist)
    # Minimize loss by updating variables in var_list.
    train_adv_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[x_star_tanh])
    # Why the last four global variables are the optimizer variables?
    # <tf.Variable 'beta1_power:0' shape=() dtype=float32_ref>
    # <tf.Variable 'beta2_power:0' shape=() dtype=float32_ref>
    # <tf.Variable 'Variable/Adam:0' shape=(10, 28, 28, 1) dtype=float32_ref>
    # <tf.Variable 'Variable/Adam_1:0' shape=(10, 28, 28, 1) dtype=float32_ref>
    optimizer_variables = tf.global_variables()[-4:]

    # The result is optimized for uint8.  Added by Weilin.
    if uint8_optimized:
        predictions = tf.argmax(model_logits(x_star_uint8), 1)
    else:
        predictions = tf.argmax(model_logits(x_star), 1)

    if targeted is False:
        correct_prediction = tf.equal(predictions, tf.argmax(y, 1))
    else:
        correct_prediction = tf.not_equal(predictions, tf.argmax(y, 1))

    # Initialize loss coefficients
    coeff_block_log = np.tile([[initial_const], [float('nan')], [float('nan')]], (1, N0))
    coeff_curr_log = coeff_block_log[0]
    coeff_high_log = coeff_block_log[1]
    coeff_low_log = coeff_block_log[2]

    # Collect best adversarial images
    best_l2 = np.zeros((N0,)) + float('nan')
    best_coeff_log = np.zeros((N0,)) + float('nan')
    best_iter = np.zeros((N0,)) + float('nan')
    best_images = np.copy(batch_images)

    # I didn't find the initialization of random perturbations?
    for _ in range(binary_search_steps):
        # Reset x_star_tanh and optimizer
        sess.run(tf.variables_initializer([x_star_tanh] + optimizer_variables))
        tf.assert_variables_initialized()

        print (coeff_curr_log) # %%%
        curr_coeff = np.exp(coeff_curr_log)
        # Initially, all are failed adversarial examples.
        all_fail = np.ones((N0,), dtype=np.bool)

        # Training loop
        improve_count = 0
        # 5000 iterations by default.
        for j in range(max_iterations):
            # Correct prediction means it is failed untargeted attacks.
            xst, adv_fail, l1o, l2d, _ = sess.run([x_star, correct_prediction, detector_penalty, l2dist, train_adv_step], feed_dict={
                x: batch_images,
                y: batch_labels,
                coeff: curr_coeff,
            })
            all_fail = np.logical_and(all_fail, adv_fail)
            for i in range(N0):
                if adv_fail[i] or l1o[i] > 0:
                    continue
                # Save the best sucessful adversarial examples, with lowest L2.
                if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
                    best_l2[i] = l2d[i]
                    best_coeff_log[i] = coeff_curr_log[i]
                    best_iter[i] = j
                    best_images[i] = xst[i]
                    improve_count += 1
            if j % 100 == 0:
                print("Adv. training iter. {}/{} improved {}".format(j, max_iterations, improve_count))
                improve_count = 0

        xst, adv_fail, l1o, l2d = sess.run([x_star, correct_prediction, detector_penalty, l2dist], feed_dict={
            x: batch_images,
            y: batch_labels,
        })
        # Run it once more, becase the last iteration in for loop doesn't get evaluated.
        for i in range(N0):
            if adv_fail[i] or l1o[i] > 0:
                continue
            if math.isnan(best_l2[i]) or l2d[i] < best_l2[i]:
                best_l2[i] = l2d[i]
                best_coeff_log[i] = coeff_curr_log[i]
                best_iter[i] = max_iterations
                best_images[i] = xst[i]
                improve_count += 1
        print("Finished training {}/{} improved {}".format(max_iterations, max_iterations, improve_count))

        # Save generated examples and their coefficients
        np.save(eval_dir + '/combined_adv_imgs.npy', best_images)
        np.save(eval_dir + '/combined_adv_coeff_log.npy', best_coeff_log)

        # Update coeff
        for i, (fail, curr, high, low) in enumerate(zip(adv_fail, coeff_curr_log, coeff_high_log, coeff_low_log)):
            if fail:
                # increase to allow more distortion
                coeff_low_log[i] = low = curr
                if math.isnan(high):
                    coeff_curr_log[i] = curr + 2.3
                else:
                    coeff_curr_log[i] = (high + low) / 2
            else:
                # decrease to penalize distortion
                coeff_high_log[i] = high = curr
                if math.isnan(low):
                    coeff_curr_log[i] = curr - 0.69
                else:
                    coeff_curr_log[i] = (high + low) / 2
        np.save(eval_dir + '/combined_coeff_log.npy', coeff_block_log)

    return best_images




