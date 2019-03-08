"""
read_train_data turns the csv output of ./preprocessMLP.C into numpy arrays to toss at models
"""
import csv
import numpy as np
import os

xdata_in = []
ydata_in = []

csvpath = "./csv/psAndPis-0-200/"
for ii, filename in enumerate(os.listdir(csvpath)):
    print("on file:", ii)
    filepath = csvpath + filename
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            xdata_row = []
            for ii, rr in enumerate(row):
                if ii == 0:
                    if int(rr) > 7:
                        continue
                    class_vec = [0,0,0,0,0,0,0,0]
                    class_vec[int(rr)] = 1
                    ydata_in.append(class_vec)
                else:
                    xdata_row.append(rr)
            xdata_in.append(xdata_row)

xdata = np.ndarray((len(xdata_in),len(xdata_in[0])))
print(len(xdata_in), len(xdata_in[0]))
for ii, xx in enumerate(xdata_in):
    for jj, yy in enumerate(xx):
        if ii>=len(xdata_in) or jj>=len(xdata_in[0]):
            continue
        xdata[ii][jj] = yy

ydata = np.ndarray((len(ydata_in),len(ydata_in[0])))

for ii, xx in enumerate(ydata_in):
    for jj, yy in enumerate(xx):
        if ii>=len(ydata_in) or jj>=len(ydata_in[0]):
            continue
        ydata[ii][jj] = yy

np.save('X_train.npy', xdata)
np.save('y_train.npy', ydata)
