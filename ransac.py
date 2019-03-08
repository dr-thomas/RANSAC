import numpy as np
from sklearn import linear_model

class ransac:
    """
    for pillaging events for tracks and other booty
    """

    def __init__(self):
        self.unused_hits = []
        self.n_ransacs = 0

    def set_data(self):
        self.X_in = np.load("./test_X_data.npy")
        self.y_in = np.load("./test_y_data.npy")
        self.hit_indecies_in = np.ndarray(len(self.X_in))
        for ii in range(len(self.hit_indecies_in)):
            self.hit_indecies_in[ii] = ii

    def ransack(self):
        if self.n_ransacs == 0:
            #initialize data here, TODO: might need to modify this to ransac unused hits after first pass...
            self.X = self.X_in
            self.y = self.y_in
            self.hit_indecies = self.hit_indecies_in
        self.n_ransacs += 1

        this_ransac = linear_model.RANSACRegressor()
        this_ransac.fit(self.X, self.y)
        inlier_mask = this_ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        ninlier = 0
        for xx in inlier_mask:
            if xx:
                ninlier += 1
        if (len(inlier_mask) - ninlier) < 4 or self.n_ransacs>10:
            #save track hypothesis
            return

        for xx in self.hit_indecies[outlier_mask]:
            self.unused_hits.append(xx)

        self.X = self.X[inlier_mask]
        self.y = self.y[inlier_mask]
        self.hit_indecies = self.hit_indecies[inlier_mask]
        self.ransack()

    #TODO
    """
      -list of tracks
      -list of hits in the event with indecies
      -method to set hits
      -method to get tracks
      -method to compare tracks in different planes for the end result of assigning hits to particle objects
      -ransac function that will iteratively find all track hypotheses in a specifed plane
    """

#class ransaced_track:
    """
    2D track found while ransacing the event
    """

    #TODO
    """
      -which plane it came from
      -list of indecies for hits that belong to it
      -score of some sort for how good of a track it is
        - score of how well the linear fit is (r^2? something like that)
    """

#class hit:

    #TODO
    """
      -x, y, z, sig, etc
    """
