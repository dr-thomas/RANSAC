import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac

def draw_ransack(viking):
    n_ransacked_tracks = len(viking.ransacked_tracks)
    #draw first 7 ransacked tracks
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


filepath = "./csv/train_0007.csv"

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

        if len(x) < 5:
            continue

        plt.figure(figsize=(15,5))

        viking = ransac.viking()
        viking.set_data(x,y)
        viking.ransack()
        plt.subplot(131)
        plt.xlabel("X")
        plt.ylabel("Y")
        draw_ransack(viking)

        viking = ransac.viking()
        viking.set_data(x,z)
        viking.ransack()
        plt.subplot(132)
        plt_title_str = str(n_true_protons) + " true protons"
        plt.title(plt_title_str)
        plt.xlabel("X")
        plt.ylabel("Z")
        draw_ransack(viking)

        viking = ransac.viking()
        viking.set_data(y,z)
        viking.ransack()
        plt.subplot(133)
        plt.xlabel("Y")
        plt.ylabel("Z")
        draw_ransack(viking)

        plt.tight_layout()

        print_string = "./png/RANSAC_test_" + str(line_count) + ".png"
        plt.savefig(print_string)
        plt.close('all')

        x_data.clear()
        y_data.clear()
        z_data.clear()
        line_count += 1

