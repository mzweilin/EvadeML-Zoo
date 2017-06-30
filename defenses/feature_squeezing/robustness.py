from .squeeze import reduce_precision_np, median_filter_np
from .squeeze import adaptive_binarize, otsu_binarize

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.output import write_to_csv

import functools, operator

def calculate_squeezed_accuracy(model, Y, X, X_adv, output_csv_fpath):
    to_csv_1 = calculate_median_smoothed_accuracy(model, Y, X, X_adv)
    to_csv_2 = calculate_color_depth_reduced_accuracy(model, Y, X, X_adv)
    #to_csv_3 = calculate_opencv_adaptive_binary_accuracy(model, Y, X, X_adv)
    #to_csv_4 = calculate_opencv_otsu_binary_accuracy(model, Y, X, X_adv)

    to_csv_list = [to_csv_1, to_csv_2]#, to_csv_3, to_csv_4]

    field_names = list(set(functools.reduce(operator.add, map(lambda x: x[0].keys(), to_csv_list))))
    records = functools.reduce(operator.add, to_csv_list)

    write_to_csv(records, output_csv_fpath, field_names)


def calculate_median_smoothed_accuracy(model, Y, X, X_adv):
    to_csv = []

    for width in range(1, 11):
        height = width
        # for height in range(1, 11):
        X_squeezed = median_filter_np(X, width, height)
        X_adv_squeezed = median_filter_np(X_adv, width, height)

        _,accuracy_leg = model.evaluate(X_squeezed, Y, batch_size=128)
        _,accuracy_adv = model.evaluate(X_adv_squeezed, Y, batch_size=128)

        to_csv.append({'width': width, 'height': height, 'accuracy_leg': accuracy_leg, 'accuracy_adv': accuracy_adv})
    return to_csv


def calculate_color_depth_reduced_accuracy(model, Y, X, X_adv):
    to_csv = []

    for npp in [256, 128, 64, 32, 16, 8, 4, 2]:
        X_squeezed = reduce_precision_np(X, npp)
        X_adv_squeezed = reduce_precision_np(X_adv, npp)

        _,accuracy_leg = model.evaluate(X_squeezed, Y, batch_size=128)
        _,accuracy_adv = model.evaluate(X_adv_squeezed, Y, batch_size=128)

        to_csv.append({'npp': npp,
                       'accuracy_leg': accuracy_leg,
                       'accuracy_adv': accuracy_adv,
                       })
    return to_csv

def calculate_opencv_adaptive_binary_accuracy(model, Y, X, X_adv):
    to_csv = []

    for block_size in [3, 5, 7, 9]:
        X_squeezed = adaptive_binarize(X, block_size=block_size)
        X_adv_squeezed = adaptive_binarize(X_adv, block_size=block_size)

        _,accuracy_leg = model.evaluate(X_squeezed, Y, batch_size=128)
        _,accuracy_adv = model.evaluate(X_adv_squeezed, Y, batch_size=128)

        to_csv.append({'block_size': block_size,
                       'accuracy_leg': accuracy_leg,
                       'accuracy_adv': accuracy_adv,
                       })
    return to_csv

def calculate_opencv_otsu_binary_accuracy(model, Y, X, X_adv):
    to_csv = []

    X_squeezed = adaptive_binarize(X, block_size=block_size)
    X_adv_squeezed = adaptive_binarize(X_adv, block_size=block_size)

    _,accuracy_leg = model.evaluate(X_squeezed, Y, batch_size=128)
    _,accuracy_adv = model.evaluate(X_adv_squeezed, Y, batch_size=128)

    to_csv.append({'otsu': 1,
                   'accuracy_leg': accuracy_leg,
                   'accuracy_adv': accuracy_adv,
                   })
    return to_csv