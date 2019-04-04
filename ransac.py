import numpy as np
from sklearn import linear_model
import math
import statistics as stats

#General TODO:
"""
  - should optimize arrays and things for computational eff at some point

  - collect all parameters and have a set fucntion:
    - cos() matching tolerance for clean
    - RANSAC params, currently using only stop prob
    - number of ouliers found stopping criteriea for ransacking
    - number of ransacks or min number of points where to stop ransacking
"""

class ransacked_track:
    """
    2D track found while ransacing the event
    """
    #TODO:
    """
    change hit indecies to a hit mask relative to the original indecies?
    """
    def __init__(self, in_hits, slope, intercept):
        self.hit_indecies = []
        for xx in in_hits:
            self.hit_indecies.append(xx)
        self.slope = slope
        self.intercept = intercept
        self.compare_n_hits = len(self.hit_indecies)
    def add_track(self, in_track):
        if in_track.compare_n_hits > self.compare_n_hits:
            self.slope = in_track.slope
            self.intercept = in_track.intercept
            self.compare_n_hits = in_track.compare_n_hits
        for xx in in_track.hit_indecies:
            self.hit_indecies.append(xx)
    def add_hit(self, index):
        self.hit_indecies.append(index)

class viking:
    """
    for pillaging events for tracks like clusters
    """

    def __init__(self, label):
        self.unused_hits = []
        self.n_ransacs = 0
        self.ransacked_tracks = []
        self.label = label

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
        self.x_min = 555e10
        for xx in self.X_in:
            if xx[0] < self.x_min:
                self.x_min = xx[0]
        self.y_min = 555e10
        for xx in self.y_in:
            if xx < self.y_min:
                self.y_min = xx
        for ii in range(len(self.X_in)):
            self.X_in[ii][0] = self.X_in[ii][0] - self.x_min
        for ii in range(len(self.y_in)):
            self.y_in[ii] = self.y_in[ii] - self.y_min

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
            this_track = ransacked_track(self.hit_indecies, this_ransac.estimator_.coef_[0], 
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

    def clean_tracks(self):
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
                if abs(cos(tracks[ii], tracks[jj])) > 0.85:
                    this_track.add_track(that_track)
                    used_tracks.append(jj)
            out_tracks.append(this_track)
        self.ransacked_tracks = out_tracks

    def grow_tracks(self):
        evt_closest_indecies = []
        for ii in range(len(self.X_in)):
            x = self.X_in[ii][0]
            y = self.y_in[ii] 
            min_dist = 555e10
            min_index = -1
            for ii, xx in enumerate(self.ransacked_tracks):
                a = xx.slope
                b = xx.intercept
                dist = abs(a*x-y+b)/(math.sqrt(a*a+1))
                if dist < min_dist:
                    min_dist = dist
                    min_index = ii
            evt_closest_indecies.append(min_index)
            for ii in range(len(self.ransacked_tracks)):
                self.ransacked_tracks[ii].hit_indecies.clear()
            for ii, xx in enumerate(evt_closest_indecies):
                if xx == -1:
                    continue
                self.ransacked_tracks[int(xx)].add_hit(ii)

    def get_track_indecies(self):
        evt_indecies = [-1 for ii in range(len(self.X_in))]
        for ii in range(len(self.X_in)):
            for itrack, track in enumerate(self.ransacked_tracks):
                found_in_track = False
                for hit in track.hit_indecies:
                    if hit == ii:
                        found_in_track = True
                        evt_indecies[ii] = itrack
                        break
                    if found_in_track:
                        break
        return evt_indecies

    def find_vertex_2D(self):
        vtxs = []
        for ii, track1 in enumerate(self.ransacked_tracks):
            for jj, track2 in enumerate(self.ransacked_tracks):
                if ii == jj:
                    continue
                m1 = track1.slope
                b1 = track1.intercept
                m2 = track2.slope
                b2 = track2.intercept
                if abs(m1-m2) < 1e-3:
                    continue
                vtx_x = (b2-b1)/(m1-m2)
                vtx_y = m1*vtx_x+b1
                vtxs.append([vtx_x,vtx_y,(track1.compare_n_hits + track2.compare_n_hits)])
        #TODO: only works if data has been scaled (data really should have been scaled by now anyway),
        #      should check that it has been with a flag or something
        self.found_vertex = False
        self.vertex_2D = [-555e10,-555e10]

        if len(vtxs) == 0:
            return
        else:
            self.found_vertex = True

        if len(vtxs) == 1:
            self.vertex_2D[0] = vtxs[0][0] + self.x_min
            self.vertex_2D[1] = vtxs[0][1] + self.y_min
            return

        x_max = 0
        for xx in self.X_in:
            if xx[0] > x_max:
                x_max = xx[0]
        y_max = 0
        for xx in self.y_in:
            if xx > y_max:
                y_max = xx

        vtx_x_mean = stats.mean([xx[0] for xx in vtxs])
        vtx_x_std = stats.stdev([xx[0] for xx in vtxs])
        vtx_y_mean = stats.mean([xx[1] for xx in vtxs])
        vtx_y_std = stats.stdev([xx[1] for xx in vtxs])

        if vtx_x_std > x_max/4. or vtx_y_std > y_max/4.:
            n_max = -555e10
            imax_vtx = -1
            for ii, vv in enumerate(vtxs):
                if n_max > vv[2]:
                    n_max = vv[2]
                    imax_vtx = ii
            self.vertex_2D[0] = vtxs[imax_vtx][0] + self.x_min
            self.vertex_2D[1] = vtxs[imax_vtx][1] + self.y_min
        else:
            self.vertex_2D[0] = vtx_x_mean + self.x_min
            self.vertex_2D[1] = vtx_y_mean + self.y_min

    def split_colinear_tracks(self, vtx):
        track_indecies_to_delete = []
        new_tracks = []
        for itrack, track in enumerate(self.ransacked_tracks):
            a = track.slope
            b = track.intercept
            x = -555e10
            y = -555e10
            labels = ['X', 'Y', 'Z']
            for ilabel, ll in enumerate(labels):
                if self.label[0] == ll:
                    x = vtx[ilabel] - self.x_min
                if self.label[1] == ll:
                    y = vtx[ilabel] - self.y_min
            dist = abs(a*x-y+b)/(math.sqrt(a*a+1))
            if dist < 5.:
                track_indecies_to_delete.append(itrack)
                track1 = ransacked_track([], track.slope, track.intercept)
                track2 = ransacked_track([], track.slope, track.intercept)
                for hit in track.hit_indecies:
                    if self.X_in[hit][0] < x:
                        track1.add_hit(hit)
                    else:
                        track2.add_hit(hit)
                new_tracks.append(track1)
                new_tracks.append(track2)
        track_indecies_to_delete.sort(reverse=True)
        for del_index in track_indecies_to_delete:
            del self.ransacked_tracks[int(del_index)]
        for track in new_tracks:
            self.ransacked_tracks.append(track)

