import numpy as np


def estimateF(x1, x2):
    """
    :param x1: Points from image 1, with shape (coordinates, point_id)
    :param x2: Points from image 2, with shape (coordinates, point_id)
    :return F: Estimated fundamental matrix
    """

    # Use x1 and x2 to construct the equation for homogeneous linear system
    ##-your-code-starts-here-##
    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, x1.shape[1]))))
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, x2.shape[1]))))
    ##-your-code-ends-here-##

    # Use SVD to find the solution for this homogeneous linear system by
    # extracting the row from V corresponding to the smallest singular value.
    ##-your-code-starts-here-##
    A = np.array([x2[0, :] * x1[0, :], x2[0, :] * x1[1, :], x2[0, :],
                  x2[1, :] * x1[0, :], x2[1, :] * x1[1, :], x2[1, :],
                  x1[0, :], x1[1, :], np.ones(x1.shape[1])]).T

    ##-your-code-ends-here-##
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)  # reshape to acquire Fundamental matrix F
    

    # Enforce constraint that fundamental matrix has rank 2 by performing
    # SVD and then reconstructing with only the two largest singular values
    # Reconstruction is done with u @ s @ vh where s is the singular values
    # in a diagonal form.
    ##-your-code-starts-here-##

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    ##-your-code-ends-here-##
    
    return F
