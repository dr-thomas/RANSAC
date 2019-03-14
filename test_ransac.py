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
    for ii in range(len(x_draw)):
        x_draw[ii] = viking.X_in[int(unused_hits[ii])][0]
    for ii in range(len(y_draw)):
        y_draw[ii] = viking.y_in[int(unused_hits[ii])]
    plt.scatter(x_draw, y_draw, color='k', marker='.')

    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    if clean:
        viking.clean_tracks()
    if grow:
        viking.grow_tracks()
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

def cluster_hits_from_ransack(vikings, ievent, x, y, z):
    #TODO
    """
      - combine confused clusters:
        -> to nearest 'good' cluster?  

      - identify 'high confidence clusters' as clusters that have high overlap across views
        - if a cluster is seen to be an intersection of two high confidence clusters, split hits into nearer high confidence
          cluster 

      - form 3-D tracks from high overlap or otherwise high confidence 3D clusters, assign all remaining hits to their
        nearest track in 3D

     DO THIS ONE:
      - for all possible 3D clusters (what is built now), check for consisteny amoung planes: compute 3 3D tracks by 
        comparing across each possible pair of planes.  form 3D tracks from conistent ones, assign hits from all 
        inconsistent clusters to nearest 3D track
    """

    cluster_X = np.ndarray((len(x),len(vikings)))

    for ii, viking in enumerate(vikings):
        track_indecies = viking.get_track_indecies()
        for jj, index in enumerate(track_indecies):
            cluster_X[jj][ii] = index[0]

    clusters = []
    for xx in cluster_X:
        cluster = 10000*xx[0] + 100*xx[1] + xx[2]
        seen_before = False
        for cc in clusters:
            if cc == cluster:
                seen_before = True
                break
        if not seen_before:
            clusters.append(cluster)

    clusters_count = [0 for ii in range(len(clusters))]
    for xx in cluster_X:
        cluster = 10000*xx[0] + 100*xx[1] + xx[2]
        for ii, cc in enumerate(clusters):
            if cluster == cc:
                clusters_count[ii] += 1
                break


    evt_labels = []
    for xx in cluster_X:
        cluster = 10000*xx[0] + 100*xx[1] + xx[2]
        for ii, cc in enumerate(clusters):
            if cluster == cc:
                if clusters_count[ii] < 5:
                    evt_labels.append(-1)
                else:
                    evt_labels.append(ii)
                break

    n_clusters = -1
    for xx in evt_labels:
        if xx > n_clusters:
            n_clusters = xx
    n_clusters += 1

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    plt.subplot(337)
    for ii in range(len(x)):
        cluster = evt_labels[ii]
        if cluster == -1:
            continue
        plt.scatter(x[ii], y[ii], color=colors[cluster%6], marker='.')
    plt.subplot(338)
    plt_title_str = str(n_clusters) + " clusters"
    plt.title(plt_title_str)
    for ii in range(len(x)):
        cluster = evt_labels[ii]
        if cluster == -1:
            continue
        plt.scatter(x[ii], z[ii], color=colors[cluster%6], marker='.')
    plt.subplot(339)
    for ii in range(len(x)):
        cluster = evt_labels[ii]
        if cluster == -1:
            continue
        plt.scatter(y[ii], z[ii], color=colors[cluster%6], marker='.')

"""
    print_string = "./png/DBSCAN_test_" + str(line_count) + ".png"
    plt.savefig(print_string)
    plt.close('all')
    """



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

        plt.figure(figsize=(15,15))

        vikings = []

        viking = ransac.viking()
        viking.set_data(x,y)
        viking.scale_data()
        viking.ransack()
        plt.subplot(331)
        plt.xlabel("X")
        plt.ylabel("Y")
        draw_ransack(viking,False, False)
        plt.subplot(334)
        plt.xlabel("X")
        plt.ylabel("Y")
        draw_ransack(viking,True, True)


        vikings.append(viking)

        viking = ransac.viking()
        viking.set_data(x,z)
        viking.scale_data()
        viking.ransack()
        plt.subplot(332)
        plt_title_str = str(n_true_protons) + " true protons"
        plt.title(plt_title_str)
        plt.xlabel("X")
        plt.ylabel("Z")
        draw_ransack(viking,False, False)
        plt.subplot(335)
        plt.xlabel("X")
        plt.ylabel("Z")
        draw_ransack(viking,True, True)


        vikings.append(viking)

        viking = ransac.viking()
        viking.set_data(y,z)
        viking.scale_data()
        viking.ransack()
        plt.subplot(333)
        plt.xlabel("Y")
        plt.ylabel("Z")
        draw_ransack(viking,False, False)
        plt.subplot(336)
        plt.xlabel("Y")
        plt.ylabel("Z")
        draw_ransack(viking,True, True)


        vikings.append(viking)

        cluster_hits_from_ransack(vikings, line_count, x, y, z)

        plt.tight_layout()

        print_string = "./png/RANSAC_test_" + str(line_count) + ".png"
        plt.savefig(print_string)
        plt.close('all')


        x_data.clear()
        y_data.clear()
        z_data.clear()
        line_count += 1

