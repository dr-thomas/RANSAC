import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac
from sklearn.cluster import DBSCAN
from sklearn import metrics

filepath = "./csv/train_0007.csv"

x_data = []
y_data = []
z_data = []
with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    n_true_protons = 0
    n_features = -1
    for iline, row in enumerate(csv_reader):
        n_features = len(row)
        n_features += -1
        print("on event:", iline)
        for ii, xx in enumerate(row):
            if ii == 0:
                n_true_protons = xx
            elif ((ii-1)%4) == 0:
                x_data.append(float(xx))
            elif ((ii-1)%4) == 1:
                y_data.append(float(xx))
            elif ((ii-1)%4) == 2:
                z_data.append(float(xx))

        hit_data = [[],[],[]]
        for ii in range(len(x_data)):
            if not (x_data[ii] == 0 and y_data[ii] == 0 and z_data[ii] == 0):
                hit_data[0].append(x_data[ii])
                hit_data[1].append(y_data[ii])
                hit_data[2].append(z_data[ii])

        if len(hit_data[0]) < 5:
            continue

        vikings = []
        for ii in range(len(hit_data)):
            for jj in range(len(hit_data)):
                viking = ransac.viking()
                viking.set_data(hit_data[ii],hit_data[jj])
                viking.scale_data()
                viking.ransack()
                vikings.append(viking)

        evt_labels = ransac.cluster_hits(vikings,hit_data)
        n_clusters = -1
        for ee in evt_labels:
            if ee > n_clusters:
                n_clusters = ee
        n_clusters += 1

        clusters = [ [] for ii in range(n_clusters)]
        for ievt, ee in enumerate(evt_labels):
            if ee < 0:
                continue
            for ii in range(3):
                clusters[int(ee)].append(hit_data[ii][ievt])

        n_features = 1000
        for ii in range(len(clusters)):
            for jj in range(len(clusters[ii]), n_features):
                clusters[ii].append(0)

        #TODO: don't forget to write truth information back in here or account for it some other way
        #TODO: will need to seperate by evt as well likely for actual use
        #TODO: should re-create file somewhere upstairs to avoid accidental appends to exisiting old files
        with open('test.csv', mode='a') as outf:
            out_writer = csv.writer(outf, delimiter=',',)
            for cc in clusters:
                out_writer.writerow(cc)

        for ii in range(len(hit_data)):
            hit_data[ii].clear()

        x_data.clear()
        y_data.clear()
        z_data.clear()
