import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac
from sklearn.cluster import DBSCAN
from sklearn import metrics

def draw_ransack(viking, clean):

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
    """
       for each track in each ransack, if some fraction of hit indecies are shared by another track in event, 
       take interesection of tracks? 

       for tracks for vikings, take all possible interesections and grab so many biggest ones? 
          -> not this

       count degenerate tracks across all views, if say 90% of the hits in a track are also in a track in a different view,
       combine those tracks (union or intersect? something else?), more generally if tracks are similar by some 
       measure, then combine.

       for each point, form a vector whose elements are the deistances to each line -> do DBSCAN or something like 
       that on these features
         -> might not need grow_tracks method wiht this idea
         -> also might not need to clean tracks
    """
    cluster_X_t = []
    for viking in vikings:
        track_distances = viking.get_distances()
        for dd in track_distances:
            cluster_X_t.append(dd)
    if len(cluster_X_t) < 1:
        return

    cluster_X = np.ndarray((len(cluster_X_t[0]),len(cluster_X_t)))

    for ii in range(len(cluster_X_t)):
        for jj in range(len(cluster_X_t[ii])):
            cluster_X[jj][ii] = cluster_X_t[ii][jj]

    db = DBSCAN(eps=0.1, min_samples=2, metric='cosine').fit(cluster_X)

    n_clusters = -1
    for xx in db.labels_:
        if xx > n_clusters:
            n_clusters = xx
    n_clusters += 1

    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    plt.subplot(337)
    for ii in range(len(x)):
        db_class = db.labels_[ii]
        plt.scatter(x[ii], y[ii], color=colors[db_class%6], marker='.')
    plt.subplot(338)
    plt_title_str = str(n_clusters) + " clusters"
    plt.title(plt_title_str)
    for ii in range(len(x)):
        db_class = db.labels_[ii]
        plt.scatter(x[ii], z[ii], color=colors[db_class%6], marker='.')
    plt.subplot(339)
    for ii in range(len(x)):
        db_class = db.labels_[ii]
        plt.scatter(y[ii], z[ii], color=colors[db_class%6], marker='.')

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
        draw_ransack(viking,False)
        plt.subplot(334)
        plt.xlabel("X")
        plt.ylabel("Y")
        draw_ransack(viking,True)

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
        draw_ransack(viking,False)
        plt.subplot(335)
        plt.xlabel("X")
        plt.ylabel("Z")
        draw_ransack(viking,True)

        vikings.append(viking)

        viking = ransac.viking()
        viking.set_data(y,z)
        viking.scale_data()
        viking.ransack()
        plt.subplot(333)
        plt.xlabel("Y")
        plt.ylabel("Z")
        draw_ransack(viking,False)
        plt.subplot(336)
        plt.xlabel("Y")
        plt.ylabel("Z")
        draw_ransack(viking,True)

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

