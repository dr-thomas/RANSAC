import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac

#TODO:
"""
  - grab z-data simultatneously below, then do all three ransacks
  - does ransac work out of the box in 3-D?
"""

filepath = "./csv/train_0004.csv"

x_data = []
y_data = []
z_data = []
with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    n_true_protons = 0
    for row in csv_reader:
        used_row = True
        for ii, xx in enumerate(row):
            if ii == 0:
                if int(xx) < 2:
                    used_row = False
                    break
                else:
                    print("event:", line_count, "has", xx, "true protons in it.")
                    n_true_protons = xx
            if ((ii-1)%4) == 0:
                x_data.append(float(xx))
            elif ((ii-1)%4) == 1:
                y_data.append(float(xx))
            elif ((ii-1)%4) == 2:
                z_data.append(float(xx))
        if not used_row:
            continue

        x = []
        y = []
        z = []
        for ii in range(len(x_data)):
            if not (x_data[ii] == 0 and y_data[ii] == 0 and z_data[ii] == 0):
                x.append(x_data[ii])
                y.append(y_data[ii])
                z.append(z_data[ii])

        viking = ransac.viking()

        viking.set_data(x,y)

        viking.ransack()

        print("number of ransacked tracks:", len(viking.ransacked_tracks))

        for ii in range(len(viking.ransacked_tracks)):
            print("track:", ii, "has", len(viking.ransacked_tracks[ii].hit_indecies), "hits")

        n_ransacked_tracks = len(viking.ransacked_tracks)
        #if n_ransacked_tracks > 7:
            #n_ransacked_tracks = 7

#draw first 7 ransacked tracks
        plt.figure()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for itrack in range(n_ransacked_tracks):
            x_draw = np.ndarray(len(viking.ransacked_tracks[itrack].hit_indecies))
            y_draw = np.ndarray(len(viking.ransacked_tracks[itrack].hit_indecies))

            for ii in range(len(x_draw)):
                x_draw[ii] = viking.X_in[int(viking.ransacked_tracks[itrack].hit_indecies[ii])][0]
            for ii in range(len(y_draw)):
                y_draw[ii] = viking.y_in[int(viking.ransacked_tracks[itrack].hit_indecies[ii])]

            plt.scatter(x_draw, y_draw, color=colors[itrack%7], marker='.')
            a = viking.ransacked_tracks[itrack].slope
            b = viking.ransacked_tracks[itrack].intercept
            plt.plot([x_draw.min(), x_draw.max()], [a*x_draw.min()+b, a*x_draw.max()+b], color=colors[itrack%7])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt_title_str = str(n_true_protons) + " true protons"
        plt.title(plt_title_str)
        print_string = "./png/RANSAC_test_" + str(line_count) + ".png"
        plt.savefig(print_string)
        plt.close('all')

        x_data.clear()
        y_data.clear()
        line_count += 1

