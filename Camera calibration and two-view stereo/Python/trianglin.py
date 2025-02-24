import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##
    A = np.vstack([
        x1[0]*P1[2, :]-P1[0, :],
        x1[1]*P1[2, :]-P1[1, :],
        x2[0]*P2[2, :]-P2[0, :],
        x2[1]*P2[2, :]-P2[1, :]
    ])
    
    # Solve X
    ATA = A.T @ A
    values, vectors = np.linalg.eig(ATA)
    X = vectors[:, np.argmin(values)]
    X = X/X[-1]
    ##-your-code-ends-here-##

    
    return X
