import numpy as np
import matplotlib.pyplot as plt
from warnings import warn


def draw_eplines(eplines, im, style='c-'):
    """
    :param eplines:
    :param im:
    :param style:
    :return:
    """
    
    for i in range(np.shape(eplines)[1]):
        a = eplines[0, i]
        b = eplines[1, i]
        c = eplines[2, i]
        
        # Draw the lines within the image borders
        if -c/b < 0:
            yc1 = 0
            xc1 = -c / a
        elif-c/b > np.shape(im)[1]:
            yc1 = np.shape(im)[1]
            xc1 = (-b*yc1 - c) / a
        else:
            xc1 = 0
            yc1 = -c / b
            
        if (-a * np.shape(im)[1] - c) / b < 0:
            yc2 = 0
            xc2 = -c/a
        elif(-a * np.shape(im)[1] - c) / b > np.shape(im)[0]:
            yc2 = np.shape(im)[0]
            xc2 = (-b*yc2-c)/a
        else:
            xc2 = np.shape(im)[1]
            yc2 = (-a*xc2 - c) / b

        plt.plot([xc1, xc2], [yc1, yc2], style, linewidth=1)
        #plt.plot([xc1 + np.shape(im1)[1], xc2 + np.shape(im1)[1]], [yc1, yc2], style, linewidth=1)


def normalise2dpts(pts):
    if np.shape(pts)[0] != 3:
        raise ValueError('pts must be 3xN')

    # Find the indicies of the points that are not at infinity
    finiteind = np.nonzero(np.absolute(pts[2,:]) > 2.2204e-16)[0]
    
    if np.size(finiteind) != np.shape(pts)[1]:
        warn('Some points are at infinity')

    # For the finite points ensure homogeneous coords have scale of 1
    pts[0, finiteind] = pts[0, finiteind] / pts[2, finiteind]
    pts[1, finiteind] = pts[1, finiteind] / pts[2, finiteind]
    pts[2, finiteind] = 1
    
    c = np.mean(pts[0:2, finiteind], axis=1)  # Centroid of finite points
    newp = np.zeros(np.shape(pts))
    newp[0, finiteind] = pts[0, finiteind] - c[0]  # Shift origin to centroid
    newp[1, finiteind] = pts[1, finiteind] - c[1]
    
    dist = np.sqrt(newp[0, finiteind]**2 + newp[1, finiteind]**2)
    meandist = np.mean(dist)
    
    scale = np.sqrt(2) / meandist
    
    T = np.array([[scale, 0, -scale * c[0]], 
               [0, scale, -scale * c[1]], 
               [0, 0, 1]])

    newpts = np.dot(T, pts)
    
    return newpts, T


def vgg_F_from_P(P1, P2):
    # Compute fundamental matrix from two camera matrices.
    X1 = P1[[1,2], :]
    X2 = P1[[2,0], :]
    X3 = P1[[0,1], :]
    Y1 = P2[[1,2], :]
    Y2 = P2[[2,0], :]
    Y3 = P2[[0,1], :]
    
    F = np.array([[np.linalg.det(np.vstack((X1, Y1))),
                   np.linalg.det(np.vstack((X2, Y1))),
                   np.linalg.det(np.vstack((X3, Y1)))],
               [np.linalg.det(np.vstack((X1, Y2))),
                np.linalg.det(np.vstack((X2, Y2))),
                np.linalg.det(np.vstack((X3, Y2)))],
               [np.linalg.det(np.vstack((X1, Y3))),
                np.linalg.det(np.vstack((X2, Y3))),
                np.linalg.det(np.vstack((X3, Y3)))]])
    return F