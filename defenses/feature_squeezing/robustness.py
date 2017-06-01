from squeeze import reduce_precision_np, median_filter_np

import csv

def write_to_csv(li, fpath, fieldnames):
    with open(fpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for di in li:
            writer.writerow(di)
            


def calculate_squeezed_accuracy(model, Y, X, X_adv, output_csv_fpath):
    to_csv_1 = calculate_median_smoothed_accuracy(model, Y, X, X_adv)
    to_csv_2 = calculate_color_depth_reduced_accuracy(model, Y, X, X_adv)

    field_names = list(set(to_csv_1[0].keys() + to_csv_2[0].keys())) # May need to pad the empty fields.
    write_to_csv(to_csv_1+to_csv_2, output_csv_fpath, field_names)


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
