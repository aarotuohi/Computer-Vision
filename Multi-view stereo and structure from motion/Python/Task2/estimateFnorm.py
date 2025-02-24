import numpy as np
from utils import normalise2dpts
from estimateF import estimateF


def estimateFnorm(x1, x2):
    """
    :param x1: Points from image 1, with shape (coordinates, point_id)
    :param x2: Points from image 2, with shape (coordinates, point_id)
    :return F: Estimated fundamental matrix
    """
    # Normalize each set of points so that the origin is at centroid
    # and mean distance from origin is sqrt(2).
    # normalise2dpts also ensures the scale parameter is 1.
    x1, T1 = normalise2dpts(x1)
    x2, T2 = normalise2dpts(x2)
    
    # Use eight-point algorithm to estimate fundamental matrix F
    F = estimateF(x1, x2)

    # Denormalize back to original coordinates
    ##-your-code-starts-here-##
    F = T2.T @ F @ T1
    ##-your-code-ends-here-##
    
    return F
