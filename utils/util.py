import numpy as np

def rand_unit_sphere(npoints, ndim):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center
    npoints -- number of feature vectors to generate
    ndim -- how many features per vector
    returns -- np array of shape (npoints, ndim), dtype=float64
    '''
    vec = np.random.randn(npoints, ndim)
    vec = np.divide(vec, np.expand_dims(np.linalg.norm(vec, axis=1), axis=1))
    return vec

def rand_unit_rect(npoints, ndim):
    '''
    Generates "npoints" number of vectors of size "ndim"
    such that each vectors is a point on an "ndim" dimensional sphere
    that is, so that each vector is of distance 1 from the center
    npoints -- number of feature vectors to generate
    ndim -- how many features per vector
    returns -- np array of shape (npoints, ndim), dtype=float64
    '''
    vec = np.random.randint(0, 2, size=(npoints, ndim))
    vec[vec==0] = -1
    return vec