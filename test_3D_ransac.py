import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac
from sklearn.cluster import DBSCAN
from sklearn import metrics

def draw_ransack(viking, clean, grow):

    unused_hits = viking.get_unused_hits()
    x_draw = np.ndarray(len(unused_hits))
    y_draw = np.ndarray(len(unused_hits))
    z_draw = np.ndarray(len(unused_hits))
    for ii in range(len(x_draw)):
        x_draw[ii] = viking.X_in[int(unused_hits[ii])][0]
        y_draw[ii] = viking.X_in[int(unused_hits[ii])][1]
    for ii in range(len(z_draw)):
        z_draw[ii] = viking.y_in[int(unused_hits[ii])]
    plt.scatter(x_draw, y_draw, color='k', marker='.')

    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    """
    if clean:
        viking.clean_tracks()
    if grow:
        viking.grow_tracks()
    """
    for itrack, track in enumerate(viking.get_tracks()):
        x_draw = np.ndarray(len(track.hit_indecies))
        y_draw = np.ndarray(len(track.hit_indecies))

        for ii in range(len(x_draw)):
            x_draw[ii] = viking.X_in[int(track.hit_indecies[ii])][0]
        for ii in range(len(y_draw)):
            y_draw[ii] = viking.y_in[int(track.hit_indecies[ii])]

        plt.scatter(x_draw, y_draw, color=colors[itrack%6], marker='.')
        a = track.slope
        b = track.intercept
        plt.plot([x_draw.min(), x_draw.max()], [a*x_draw.min()+b, a*x_draw.max()+b], color=colors[itrack%6])


filepath = "./csv/train_0007.csv"

x_data = []
y_data = []
z_data = []
with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    n_true_protons = 0
    for row in csv_reader:
        for ii, xx in enumerate(row):
            if ii == 0:
                n_true_protons = xx
            elif ((ii-1)%4) == 0:
                x_data.append(float(xx))
            elif ((ii-1)%4) == 1:
                y_data.append(float(xx))
            elif ((ii-1)%4) == 2:
                z_data.append(float(xx))

        x = []
        y = []
        z = []
        for ii in range(len(x_data)):
            if not (x_data[ii] == 0 and y_data[ii] == 0 and z_data[ii] == 0):
                x.append(x_data[ii])
                y.append(y_data[ii])
                z.append(z_data[ii])

        if len(x) < 5:
            continue

        plt.figure(figsize=(15,5))

        viking = ransac.viking()
        viking.set_data(x,y,z)
        #viking.scale_data()
        viking.ransack()

        ransacked_data_list = []
        for itrack, track in enumerate(viking.get_tracks()):
            for hit in track.hit_indecies:
                x = viking.X_in[int(hit)][0]
                y = viking.X_in[int(hit)][1]
                z = viking.y_in[int(hit)]
                ransacked_data_list.append([x,y,z,itrack])
        for hit in viking.get_unused_hits():
            x = viking.X_in[int(hit)][0]
            y = viking.X_in[int(hit)][1]
            z = viking.y_in[int(hit)]
            ransacked_data_list.append([x,y,z,-1])

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        f, axes = plt.subplots(1, 3, figsize=(15,5))
        for data in ransacked_data_list:
            axes[0].scatter(data[0],data[1],color=colors[int(data[3])%6], marker='.')
            axes[1].scatter(data[1],data[2],color=colors[int(data[3])%6], marker='.')
            title_str = "ntracks: " + str(len(viking.get_tracks()))
            axes[1].set_title(title_str)
            axes[2].scatter(data[2],data[0],color=colors[int(data[3])%6], marker='.')

        print_string = "./png/RANSAC_3D_test_" + str(line_count) + ".png"
        plt.savefig(print_string)
        plt.close('all')

        x_data.clear()
        y_data.clear()
        z_data.clear()
        line_count += 1

