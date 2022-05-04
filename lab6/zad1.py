import csv
with open('iris_with_errors.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    iris_with_errors = []
    errors = []
    errorsCount = [0,0,0,0]
    count = [0,0,0,0]
    sums = [0,0,0,0]
    for row in reader:

        if row[0] != '"sepal.length"':
            iris_with_errors.append(row)
            for i in range(0, 4):
                if (row[i] in [' ', "NA", "-"]) or float(row[i])<0:
                    errors.append(row)
                    errorsCount[i]+=1
                else:
                    count[i]+=1
                    sums[i]+=float(row[i])

        if row[4] not in ['"Setosa"', '"Versicolor"', '"Virginica"']:
            print(row)
            if(row[4][1]=='s'):
                row[4] = '"Setosa"'
            if(row[4][1]=='v'):
                if(row[4][-2]=='a'):
                    row[4] = '"Virginica"'
                if(row[4][-2]=='r'):
                    row[4] = '"Versicolor"'





print(errorsCount)
print(errors)
print(sums)
average = [round(sums[0]/count[0],2), round(sums[1]/count[1],2), round(sums[2]/count[2],2), round(sums[3]/count[3],2)]
print(average)

for row in iris_with_errors:
    for i in range(0, 4):
        if (row[i] in [' ', "NA", "-"]) or float(row[i]) < 0 or float(row[i])>15:
            row[i] = average[i]

print(iris_with_errors)