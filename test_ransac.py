import ransac
from matplotlib import pyplot as plt
import numpy as np

viking = ransac.viking()

viking.set_data()

viking.ransack()

print("number of ransacked tracks:", len(viking.ransacked_tracks))

for ii in range(len(viking.ransacked_tracks)):
    print("track:", ii, "has", len(viking.ransacked_tracks[ii].hit_indecies), "hits")

n_ransacked_tracks = len(viking.ransacked_tracks)
if n_ransacked_tracks > 7:
    n_ransacked_tracks = 7

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

    plt.scatter(x_draw, y_draw, color=colors[itrack], marker='.')
    a = viking.ransacked_tracks[itrack].slope
    b = viking.ransacked_tracks[itrack].intercept
    plt.plot([x_draw.min(), x_draw.max()], [a*x_draw.min()+b, a*x_draw.max()+b], color=colors[itrack])
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("./RANSAC_test.png")
