import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinates with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A 
    A = []
    for i in range(x_world.shape[0]):
        x = x_world[i, 0]
        y = x_world[i, 1]
        z = x_world[i, 2]
        x1 = x_im[i, 0]
        y1 = x_im[i, 1]
        A.append([0, 0, 0, 0, -x, -y, -z, -1, y1*x, y1*y, y1*z, y1])
        A.append([x, y, z, 1, 0, 0, 0, 0, -x1*x, -x1*y, -x1*z, -x1])
    A = np.array(A)
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ATA = A.T @ A
    values, vectors = np.linalg.eig(ATA)
    ev = vectors[:, np.argmin(values)]
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
   
    return P