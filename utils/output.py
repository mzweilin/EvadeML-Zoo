import csv

def write_to_csv(li, fpath, fieldnames):
    with open(fpath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for di in li:
            writer.writerow(di)