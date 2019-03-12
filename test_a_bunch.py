import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac

#TODO:
"""
  - grab z-data simultatneously below, then do all three ransacks
"""

filepath = "./csv/train_0004.csv"

X_data = []
y_data = []
with open(filepath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        used_row = True
        for ii, xx in enumerate(row):
            if ii == 0:
                if int(xx) < 2:
                    used_row = False
                    break
                else:
                    print("event:", line_count, "has", xx, "true protons in it.")
            if ((ii-1)%4) == 1:
                X_data.append(float(xx))
            elif ((ii-1)%4) == 2:
                y_data.append(float(xx))
        if used_row:
            n_nonzero = 0
            for ii in range(len(X_data)):
                if X_data[ii] == 0 and y_data[ii] == 0:
                    continue
                n_nonzero += 1
        else:
            continue

        X = np.ndarray((n_nonzero,1))
        y = np.ndarray(n_nonzero)
        idata = 0
        for ii in range(len(X_data)):
            if X_data[ii] == 0 and y_data[ii] == 0:
                continue 
            X[idata][0] = X_data[ii]
            y[idata] = y_data[ii] 
            idata += 1


        viking = ransac.viking()

        viking.set_data(X,y)

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
        print_string = "./png/RANSAC_test_" + str(line_count) + ".png"
        plt.savefig(print_string)
        plt.close('all')

        X_data.clear()
        y_data.clear()
        line_count += 1

