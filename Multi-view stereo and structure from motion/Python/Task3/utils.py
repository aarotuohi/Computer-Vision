import numpy as np


def vgg_contreps(X):
    # vgg_contreps  Contraction with epsilon tensor.
    #
    # B = vgg_contreps(A) is tensor obtained by contraction of A with epsilon tensor.
    # However, it works only if the argument and result fit to matrices, in particular:
    #
    # - if A is row or column 3-vector ...  B = [A]_x
    # - if A is skew-symmetric 3-by-3 matrix ... B is row 3-vector such that A = [B]_x
    # - if A is skew-symmetric 4-by-4 matrix ... then A can be interpreted as a 3D line Pluecker matrix
    #                                               skew-symmetric 4-by-4 B as its dual Pluecker matrix.
    # - if A is row 2-vector ... B = [0 1; -1 0]*A', i.e., A*B=eye(2)
    # - if A is column 2-vector ... B = A'*[0 1; -1 0], i.e., B*A=eye(2)
    #
    # It is vgg_contreps(vgg_contreps(A)) = A.

    # werner@robots.ox.ac.uk, Oct 2001    

    if np.prod(np.shape(X)) == 3:
        X = X.flatten()
        Y = np.array([[0, X[2], -X[1]], 
                   [-X[2], 0, X[0]], 
                   [X[1], -X[0], 0]])
    elif all(np.shape(X) == (1,2)):
        Y = np.dot(np.array([[0,1], [-1,0]]), X.T)
    elif all(np.shape(X) == (2,1)):
        Y = np.dot(X.T, np.array([[0,1], [-1,0]]))
    elif all(np.shape(X) == (3,3)):
        Y = np.array([X[1,2], X[2,0], X[0,1]])
    elif all(np.shape(X) == (4,4)):
        Y = np.array([[0, X[2,3], X[3,1], X[1,2]],
                   [X[3,2], 0, X[0,3], X[2,1]],  
                   [X[1,3], X[3,0], 0, X[0,1]],
                   [X[2,1], X[0,2], X[1,0], 0]])
    else:
        raise ValueError('Wrong matrix size')
        
    return Y


def vgg_X_fromxP_lin(u, P, imsize):
    #vgg_X_from_xP_lin  Estimation of 3D point from image matches and camera matrices, linear.
    #   X = vgg_X_from_xP_lin(x,P,imsize) computes projective 3D point X (column 4-vector)
    #   from its projections in K images x (2-by-K matrix) and camera matrices P (K-cell
    #   of 3-by-4 matrices). Image sizes imsize (2-by-K matrix) are needed for preconditioning.
    #   By minimizing algebraic distance.
    #
    #   See also vgg_X_from_xP_nonlin.
    
    # werner@robots.ox.ac.uk, 2003   
    
    K = len(P)
    for k in range(K):
        H = np.array([[2.0 / imsize[0, k], 0, -1],
                   [0, 2.0 / imsize[1, k], -1],
                   [0, 0, 1]])
        P[k] = np.dot(H, P[k])
        u[:,k] = (np.dot(H[0:2, 0:2], u[:,k]).T + H[0:2,2]).T
    A = np.zeros(4)
    for k in range(K):
        tmp = np.dot(vgg_contreps(np.vstack((u[:,k], 1))), P[k])
        A = np.vstack((A,tmp))
    A = np.delete(A,0,0)
    U, D, V = np.linalg.svd(A.astype(float))
    X = V[-1, :]
    
    # Get orientation right
    tmp = np.zeros(4)
    for i in range(K):
        tmp = np.vstack((tmp, P[i][2,:]))
    tmp = np.delete(tmp,0,0)    
    s = np.dot(tmp, X)
    if any(s < 0):
        X = -X

    return X


def camcalibDLT(Xworld, Xim):
    # Direct linear transformation (DLT) is an algorithm which 
    # solves a set of variables from a set of similarity relations
    N = np.shape(Xworld)[0]
    A = np.zeros((1,12))
    for i in range(N):
        tmp = np.hstack((np.zeros((4)), Xworld[i,:], -Xim[i,1]*Xworld[i,:]))
        tmp2 = np.hstack((Xworld[i,:], np.zeros(4), -Xim[i,0]*Xworld[i,:]))
        A = np.vstack((A,tmp,tmp2))
    A = np.delete(A,0,0)
        
    M = np.dot(A.T, A)
    
    u,s,v = np.linalg.svd(M)
    idmin = np.argmin(s)
    ev = v[idmin]
    P = np.reshape(ev, (3,4))
    
    return P