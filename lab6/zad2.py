import csv
with open('iris_with_errors.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    iris = []