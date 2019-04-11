import numpy as np
import csv
from RANSAC import ransac

def cluster(hit_data_in):
    xyzs_data = [[] for ii in range(4)]

    for ii, xx in enumerate(hit_data_in):
        xyzs_data[ii%4].append(xx)

    del_indecies = []
    for ii in range(len(xyzs_data[3])):
        if xyzs_data[0][ii] == 0 and xyzs_data[1][ii] == 0 and xyzs_data[2][ii] == 0 and xyzs_data[3][ii] == 0:
            del_indecies.append(ii)

    del_indecies.sort(reverse=True)

    for dd in del_indecies:
        for xyzs in xyzs_data:
            del xyzs[dd]

    if len(xyzs_data[3]) < 5:
        return []


    vikings = []
    labels = ['X', 'Y', 'Z']
    for ii in range(3):
        for jj in range(3):
            label = labels[ii] + labels[jj]
            viking = ransac.viking(label)
            viking.set_data(xyzs_data[ii],xyzs_data[jj])
            viking.scale_data()
            viking.ransack()
            vikings.append(viking)

    evt_labels = ransac.cluster_hits(vikings,xyzs_data)
    n_clusters = -1
    for ee in evt_labels:
        if ee > n_clusters:
            n_clusters = ee
    n_clusters += 1

    clusters = [ [] for ii in range(n_clusters)]
    for ievt, ee in enumerate(evt_labels):
        if ee < 0:
            continue
        for ii in range(len(xyzs_data)):
            clusters[int(ee)].append(xyzs_data[ii][ievt])

    for ii, xx in enumerate(clusters):
        origin = [0,0,0]
        for jj, yy in enumerate(xx): 
            if jj < 3:
                origin[jj] = yy
                clusters[ii][jj] = yy - origin[jj]
            else:
                if jj%4 < 3:
                    clusters[ii][jj] = yy - origin[jj%4]


    n_features = 1000
    for ii in range(len(clusters)):
        for jj in range(len(clusters[ii]), n_features):
            clusters[ii].append(0)

    return clusters
