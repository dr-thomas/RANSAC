import numpy as np
from matplotlib import pyplot as plt
import csv
import ransac
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import statistics as stats
from numpy import random

    #TODO: this is ready for a re-factor/thorough clean up

    #TODO: once vertexing stuff is in place, seperately study how well it does in some test set


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

    #get 3D vertex
    labels = ['X', 'Y', 'Z']
    vtxs = [[],[],[]]
    for viking in vikings:
        viking.find_vertex_2D()
        if viking.found_vertex:
            for ilabel, ll in enumerate(labels):
                if viking.label[0] == ll:
                    vtxs[ilabel].append(viking.vertex_2D[0])
                if viking.label[1] == ll:
                    vtxs[ilabel].append(viking.vertex_2D[1])
    vtx = [-555e10 for ii in range(3)]
    found_vtx = False
    if len(vtxs[0]) > 0 and len(vtxs[1]) > 0 and len(vtxs[2]) > 0:
        found_vtx = True
        for ii in range(3):
            vtx[ii] = stats.mean([xx for xx in vtxs[ii]])

    if found_vtx:
        for viking in vikings:
            viking.split_colinear_tracks(vtx)

    cluster_X = np.ndarray((len(x),len(vikings)))

    for ii, viking in enumerate(vikings):
        track_indecies = viking.get_track_indecies()
        for jj, index in enumerate(track_indecies):
            cluster_X[jj][ii] = index

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
        if clusters_count[ii] >= 5:
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
        if found_vtx:
            plt.plot(vtx[0], vtx[1], marker='*', markersize=20, color='gold')
    plt.subplot(338)
    plt_title_str = str(n_clusters) + " clusters"
    plt.title(plt_title_str)
    for ii in range(len(x)):
        cluster = evt_labels[ii]
        plt.scatter(z[ii], x[ii], color=colors[cluster%7], marker='.')
        if found_vtx:
            plt.plot(vtx[2], vtx[0], marker='*', markersize=20, color='gold')
    plt.subplot(339)
    for ii in range(len(x)):
        cluster = evt_labels[ii]
        plt.scatter(y[ii], z[ii], color=colors[cluster%7], marker='.')
        if found_vtx:
            plt.plot(vtx[1], vtx[2], marker='*', markersize=20, color='gold')

n_points = 50

x = []
y = []
z = []

vtx = [random.rand()*200 for ii in range(3)]
for ii in range(n_points):
    x.append(random.rand()*10+vtx[0])
    y.append(random.rand()*10+vtx[1])
    z.append(random.rand()*10+vtx[2])

plt.figure(figsize=(15,15))

vikings = []

viking = ransac.viking('XY')
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

viking = ransac.viking('ZX')
viking.set_data(z,x)
viking.scale_data()
viking.ransack()
plt.subplot(332)
plt.xlabel("Z")
plt.ylabel("X")
draw_ransack(viking,False, False)
plt.subplot(335)
plt.xlabel("Z")
plt.ylabel("X")
draw_ransack(viking,True, True)


vikings.append(viking)

viking = ransac.viking('YZ')
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

viking = ransac.viking('ZY')
viking.set_data(z,y)
viking.scale_data()
viking.ransack()
viking.clean_tracks()
viking.grow_tracks()
vikings.append(viking)

viking = ransac.viking('XZ')
viking.set_data(x,z)
viking.scale_data()
viking.ransack()
viking.clean_tracks()
viking.grow_tracks()
vikings.append(viking)

viking = ransac.viking('YX')
viking.set_data(y,x)
viking.scale_data()
viking.ransack()
viking.clean_tracks()
viking.grow_tracks()
vikings.append(viking)

cluster_hits_from_ransack(vikings, 1, x, y, z)

plt.tight_layout()
plt.subplots_adjust(left=0.06)

print_string = "./png/RANSAC_test_noise.png"
plt.savefig(print_string)
plt.close('all')

