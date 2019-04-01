import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac
from sklearn import metrics
from sklearn.linear_model import LinearRegression

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

    cluster_X = np.ndarray((len(x),len(vikings)))

    for ii, viking in enumerate(vikings):
        track_indecies = viking.get_track_indecies()
        for jj, index in enumerate(track_indecies):
            cluster_X[jj][ii] = index[0]

    clusters = []
    for xx in cluster_X:
        cluster = 0.
        for jj, yy in enumerate(xx):
            cluster += 100**float(jj)*yy
        seen_before = False
        for cc in clusters:
            if cc == cluster:
                seen_before = True
                break
        if not seen_before:
            clusters.append(cluster)

    clusters_count = [0 for ii in range(len(clusters))]
    for xx in cluster_X:
        cluster = 0.
        for jj, yy in enumerate(xx):
            cluster += 100**float(jj)*yy
        for ii, cc in enumerate(clusters):
            if cluster == cc:
                clusters_count[ii] += 1
                break

    hc_clusters = []
    for ii in range(len(clusters)):
        if clusters_count[ii] >= 10:
            hc_clusters.append(clusters[ii])


    evt_labels = []
    for xx in cluster_X:
        cluster = 0.
        for jj, yy in enumerate(xx):
            cluster += 100**float(jj)*yy
        was_hc = False
        for ii, cc in enumerate(hc_clusters):
            if cluster == cc:
                evt_labels.append(ii)
                was_hc = True
                break
        if not was_hc:
            evt_labels.append(-1)

    are_unused = True
    attempts = 0
    while are_unused and attempts < 100:
        attempts += 1
        are_unused = False
        for ii, xx in enumerate(evt_labels):
            if xx == -1:
                are_unused = True
                min_dist = 555e10
                for jj in range(len(x)):
                    dist = (x[ii]-x[jj])*(x[ii]-x[jj])
                    dist += (y[ii]-y[jj])*(y[ii]-y[jj])
                    dist += (z[ii]-z[jj])*(z[ii]-z[jj])
                    if dist < min_dist:
                        min_dist = dist
                        evt_labels[ii] = evt_labels[jj]

    #TODO: vetex clusters here 
    """
      - in each plane, lin fit each cluster, look for intersections
      - for each intersection, give a vertex candidate in that plane, (weighted by npoints in cluster?) average the 
        positions of multiple vertecies
      - to get 3D vertex, average (again, some weighted here?) the two positions across the shared planes
      - then split clusters that are on both sides of vertex
    """

    #TODO: this is ready for a re-factor/thorough clean up

    """
      - rather than a re-fit, maybe use exisiting vikings slopes and intercepts in each of all 6 planes to do this? 
    """
    #TODO: once vertexing stuff is in place, seperately study how well it does in some test set

    n_clusters = -1
    for xx in evt_labels:
        if xx > n_clusters:
            n_clusters = xx
    n_clusters += 1

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    ax = plt.subplot(337)
    plt.xlabel("X")
    plt.ylabel("Y")
    pad = 5
    plt.annotate("Final Clusters", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label, textcoords='offset points',		
            size='large', ha='right', va='center', rotation=90)

    for ii in range(len(x)):
        cluster = evt_labels[ii]
        plt.scatter(x[ii], y[ii], color=colors[cluster%7], marker='.')
    plt.subplot(338)
    plt_title_str = str(n_clusters) + " clusters"
    plt.title(plt_title_str)
    for ii in range(len(x)):
        cluster = evt_labels[ii]
        plt.scatter(z[ii], x[ii], color=colors[cluster%7], marker='.')
    plt.subplot(339)
    for ii in range(len(x)):
        cluster = evt_labels[ii]
        plt.scatter(y[ii], z[ii], color=colors[cluster%7], marker='.')


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
        ax = plt.subplot(331)
        plt.xlabel("X")
        plt.ylabel("Y")
        pad = 5
        plt.annotate("RANSAC", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',		
                size='large', ha='right', va='center', rotation=90)

        draw_ransack(viking,False, False)
        ax = plt.subplot(334)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.annotate("Clean and Grow", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',		
                size='large', ha='right', va='center', rotation=90)

        draw_ransack(viking,True, True)


        vikings.append(viking)

        viking = ransac.viking()
        viking.set_data(z,x)
        viking.scale_data()
        viking.ransack()
        plt.subplot(332)
        plt_title_str = str(n_true_protons) + " true protons"
        plt.title(plt_title_str)
        plt.xlabel("Z")
        plt.ylabel("X")
        draw_ransack(viking,False, False)
        plt.subplot(335)
        plt.xlabel("Z")
        plt.ylabel("X")
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

        viking = ransac.viking()
        viking.set_data(z,y)
        viking.scale_data()
        viking.ransack()
        viking.clean_tracks()
        viking.grow_tracks()
        vikings.append(viking)

        viking = ransac.viking()
        viking.set_data(x,z)
        viking.scale_data()
        viking.ransack()
        viking.clean_tracks()
        viking.grow_tracks()
        vikings.append(viking)

        viking = ransac.viking()
        viking.set_data(y,x)
        viking.scale_data()
        viking.ransack()
        viking.clean_tracks()
        viking.grow_tracks()
        vikings.append(viking)

        cluster_hits_from_ransack(vikings, line_count, x, y, z)

        plt.tight_layout()
        plt.subplots_adjust(left=0.06)

        print_string = "./png/RANSAC_test_" + str(line_count) + ".png"
        plt.savefig(print_string)
        plt.close('all')

        exit()

        x_data.clear()
        y_data.clear()
        z_data.clear()
        line_count += 1

