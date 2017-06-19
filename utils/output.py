import csv
import os
import sys

def disablePrint(log_fpath=None):
    sys.stdout.flush()
    if log_fpath is None:
        log_fpath = os.devnull
    sys.stdout = open(log_fpath, 'w')


def enablePrint():
    sys.stdout.flush()
    log_f = sys.stdout
    sys.stdout = sys.__stdout__
    log_f.close()


def write_to_csv(li, fpath, fieldnames):
    with open(fpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for di in li:
            writer.writerow(di)


def formatter(start, end, step):
    return '{}-{}:{}'.format(start, end, step)


def format_number_range(lst):
    """
    Format a list of numbers to a string of ranges.
    Source: https://stackoverflow.com/a/9855078
    """
    n = len(lst)
    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if lst[scan + 2] - lst[scan + 1] != step:
            result.append(str(lst[scan]))
            scan += 1
            continue

        for j in range(scan+2, n-1):
            if lst[j+1] - lst[j] != step:
                result.append(formatter(lst[scan], lst[j], step))
                scan = j+1
                break
        else:
            result.append(formatter(lst[scan], lst[-1], step))
            return ','.join(result)

    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(','.join(map(str, lst[scan:])))

    return ','.join(result)


def save_task_descriptor(result_folder, to_csv):
    task = to_csv[0]
    fname = "%s_%d_%s_%s_task_desc.csv" % (task['dataset_name'], task['test_set_selected_length'], task['test_set_selected_idx_hash'][:5], task['model_name'])
    fpath = os.path.join(result_folder, fname)

    fieldnames = ['dataset_name', 'model_name', 'accuracy_test', 'mean_confidence_test', \
                  'test_set_selected_length', 'test_set_selected_idx_ranges', 'test_set_selected_idx_hash', \
                  'accuracy_test_selected', 'mean_confidence_test_selected']
    write_to_csv(to_csv, fpath, fieldnames)
