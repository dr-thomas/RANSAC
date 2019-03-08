class ransac:
    """
    for pillaging events for tracks and other booty
    """

    #TODO
    """
      -list of tracks
      -list of hits in the event with indecies
      -method to set hits
      -method to get tracks
      -method to compare tracks in different planes for the end result of assigning hits to particle objects
      -ransac function that will iteratively find all track hypotheses in a specifed plane
    """

class ransaced_track:
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

class hit:

    #TODO
    """
      -x, y, z, sig, etc
    """
