import numpy as np
from sklearn import linear_model
import math

#General TODO:
"""
  idea: use ransac to get several lines, identify and eliminate degenerate lines by slope
        then for each point, assigin it to the line it is nearest
  Track cleaning considerations:
     - parralell tracks
     - individal tracks that have clusters separated by large distance
     - tracks that have conflicting intersections relative to agreed on vertex by other tracks???

  - should probably scale data somehow so that all events are fit in the same general x-y plane

  - far out (.. man .. ) idea: could supervise this algorithm to try to convince it more judiciously split
    hits near intersections amoung lines maybe?  would need custom implementation obviously

  - does ransac work out of the box in 3-D?

  - should optimize arrays and things for computational eff

"""

class ransacked_track:
    """
    2D track found while ransacing the event
    """
    def __init__(self, in_hits, slope, intercept):
        self.hit_indecies = []
        for xx in in_hits:
            self.hit_indecies.append(xx)
        self.slope = slope[0]
        self.intercept = intercept
        self.compare_n_hits = len(self.hit_indecies)
    def add_track(self, in_track):
        if in_track.compare_n_hits > self.compare_n_hits:
            self.slope = in_track.slope
            self.intercept = in_track.intercept
            self.compare_n_hits = in_track.compare_n_hits
        for xx in in_track.hit_indecies:
            self.hit_indecies.append(xx)

class viking:
    """
    for pillaging events for tracks like clusters
    """

    def __init__(self):
        self.unused_hits = []
        self.n_ransacs = 0
        self.ransacked_tracks = []

    def set_data(self, X_in, y_in):
        self.X_in = np.ndarray((len(X_in),1))
        self.y_in = np.ndarray(len(y_in))
        for ii, xx in enumerate(X_in):
            self.X_in[ii][0] = xx
        for ii, xx in enumerate(y_in):
            self.y_in[ii] = xx
        self.hit_indecies_in = np.ndarray(len(self.X_in))
        for ii in range(len(self.hit_indecies_in)):
            self.hit_indecies_in[ii] = ii

    def scale_data(self):
        x_min = 555e10
        for xx in self.X_in:
            if xx[0] < x_min:
                x_min = xx[0]
        y_min = 555e10
        for xx in self.y_in:
            if xx < y_min:
                y_min = xx
        for ii in range(len(self.X_in)):
            self.X_in[ii][0] = self.X_in[ii][0] - x_min
        for ii in range(len(self.y_in)):
            self.y_in[ii] = self.y_in[ii] - y_min

    def ransack(self):
        if self.n_ransacs == 0:
            self.X = self.X_in
            self.y = self.y_in
            self.hit_indecies = self.hit_indecies_in
        self.n_ransacs += 1

        #this_ransac = linear_model.RANSACRegressor(residual_threshold=2., stop_probability=0.99)
        this_ransac = linear_model.RANSACRegressor(stop_probability=0.9)
        try:
            this_ransac.fit(self.X, self.y)
        except:
            return
        inlier_mask = this_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        ninlier = 0
        for xx in inlier_mask:
            if xx:
                ninlier += 1
        if (len(inlier_mask) - ninlier) < 1:
            #save track hypothesis
            this_track = ransacked_track(self.hit_indecies, this_ransac.estimator_.coef_, 
                    this_ransac.estimator_.intercept_)
            self.ransacked_tracks.append(this_track)
            #set start ransacking again with unused hits
            self.X = np.ndarray((len(self.unused_hits),1))
            self.y = np.ndarray(len(self.unused_hits))
            self.hit_indecies = np.ndarray(len(self.unused_hits))
            for ii in range(len(self.X)):
                self.X[ii][0] = self.X_in[int(self.unused_hits[ii])][0]
            for ii in range(len(self.y)):
                self.y[ii] = self.y_in[int(self.unused_hits[ii])]
            for ii in range(len(self.hit_indecies)):
                self.hit_indecies[ii] = self.unused_hits[ii]
            self.unused_hits = []

        else:
            for xx in self.hit_indecies[outlier_mask]:
                self.unused_hits.append(xx)

            self.X = self.X[inlier_mask]
            self.y = self.y[inlier_mask]
            self.hit_indecies = self.hit_indecies[inlier_mask]

        if self.n_ransacs > 100 or len(self.X) < 5:
            return

        self.ransack()

    def get_unused_hits(self):
        self.unused_hits = [ii for ii in range(len(self.X_in))]
        for track in self.ransacked_tracks:
            for hit in track.hit_indecies:
                idel = -1
                for iuhit, uhit in enumerate(self.unused_hits):
                    if uhit == hit:
                        idel = iuhit
                        break
                if idel >= 0:
                    del self.unused_hits[idel]
        return self.unused_hits

    def get_tracks(self):
        return self.ransacked_tracks

    def get_cleaned_tracks(self):
        def cos(track1, track2):
            a1 = track1.slope
            a2 = track2.slope
            return (1+a1*a2)/(math.sqrt(1+a1*a1)*math.sqrt(1+a2*a2))
        out_tracks = []
        used_tracks = []
        tracks = self.get_tracks()
        for ii, this_track in enumerate(self.get_tracks()):
            is_used = False 
            for xx in used_tracks:
                if ii == int(xx):
                    is_used = True
                    break
            if is_used:
                continue
            used_tracks.append(ii)
            for jj, that_track in enumerate(self.get_tracks()):
                is_used = False 
                for xx in used_tracks:
                    if jj == int(xx):
                        is_used = True
                        break
                if is_used:
                    continue
                if abs(cos(tracks[ii], tracks[jj])) > 0.95:
                    this_track.add_track(that_track)
                    used_tracks.append(jj)
            out_tracks.append(this_track)
        return out_tracks





