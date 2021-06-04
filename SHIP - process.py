import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import date
from datetime import datetime
import random


# returns ndarray with the following dimensions: nProcesses, nPaths, nPoints
# https://mikejuniperhill.blogspot.com/2019/04/python-path-generator-for-correlated.html
def GeneratePaths(spot, process, maturity, nPoints, nPaths, correlation):
    dt = maturity / (nPoints-1)

    # case: given correlation matrix, create paths for multiple correlated processes
    if (isinstance(correlation, np.ndarray)):
        nProcesses = process.shape[0]
        result = np.zeros(shape=(nProcesses, nPaths, nPoints))

        # loop through number of paths
        for i in range(nPaths):
            # create one set of correlated random variates for n processes
            choleskyMatrix = np.linalg.cholesky(correlation)
            e = np.random.normal(size=(nProcesses, nPoints))
            paths = np.dot(choleskyMatrix, e)
            # loop through number of steps
            for j in range(nPoints):
                # loop through number of processes
                for k in range(nProcesses):
                    # first path value is always current spot price
                    if (j == 0):
                        result[k, i, j] = paths[k, j] = spot[k]
                    else:
                        # use SDE lambdas (inputs: previous spot, dt, current random variate)
                        result[k, i, j] = paths[k, j] = process[k](paths[k, j - 1], dt, paths[k, j])

    return result



def GBM_v2(seed, mu, sig, dim, spots, maturity, nPoints, nPaths, correlation=None):
    np.random.seed(seed)
    one_process = lambda s, dt, e: s + mu * s * dt + sig * s * np.sqrt(dt) * e
    processes = np.repeat(one_process, dim)
    x = GeneratePaths(spots, processes, maturity, nPoints, nPaths, correlation)
    x = np.reshape(x, newshape=(nPaths, dim, nPoints))
    t = np.linspace(0., maturity, int(nPoints))
    return x, t


def Am_put_geom(K, x):
    bs = tf.shape(x)[0];
    dim = tf.shape(x).numpy()[1];
    tN = tf.shape(x)[2]
    x1 = tf.math.multiply(1 / dim, tf.reduce_sum(x, axis=1))
    x2 = tf.reshape(x1, shape=(bs, tN, 1))
    g = tf.math.maximum(
        tf.math.subtract(K, x2), 0)
    return g



def GeneratePaths_onedim(spot, process, maturity, nPoints, nPaths):
    dt = maturity / (nPoints-1)

    result = np.zeros(shape=(1, nPaths, nPoints))
    # loop through number of paths
    for i in range(nPaths):
        # create one set of random variates for one process
        path = np.random.normal(size=nPoints)
        # first path value is always current spot price
        result[0, i, 0] = path[0] = spot
        # loop through number of steps
        for j in range(nPoints):
            if (j > 0):
                # use SDE lambda (inputs: previous spot, dt, current random variate)
                result[0, i, j] = path[j] = process(path[j - 1], dt, path[j])
    return result

def GeneratePaths_onedim_rand(process, maturity, nPoints, nPaths):
    dt = maturity / (nPoints-1)

    result = np.zeros(shape=(1, nPaths, nPoints))
    # loop through number of paths
    for i in range(nPaths):
        # create one set of random variates for one process
        path = np.random.normal(size=nPoints)
        # first path value is always current spot price
        result[0, i, 0] = path[0] = (0.9 - 0.1) *random.random() + 0.1 #np.round( random.random(), 1 )
        # loop through number of steps
        for j in range(nPoints):
            if (j > 0):
                # use SDE lambda (inputs: previous spot, dt, current random variate)
                result[0, i, j] = path[j] = process(path[j - 1], dt, path[j])
                #if np.maximum(path[j])
    return result



'P&Shiryaev - eq. (21.0.9)'
def PS_Hip(seed, mu, sig, p,  maturity, nPoints, nPaths, randstart = False):
    np.random.seed(seed)
    one_process = lambda s, dt, e: s + (mu/sig) * s * (1-s) * np.sqrt(dt) * e
    if randstart:
        x = GeneratePaths_onedim_rand(process=one_process,
                                 maturity=maturity, nPoints=nPoints, nPaths=nPaths)
    else:
        x = GeneratePaths_onedim(spot=p, process=one_process,
                             maturity=maturity, nPoints=nPoints, nPaths=nPaths)
    x = np.reshape(x, newshape=(nPaths, nPoints))
    t = np.linspace(0, maturity, nPoints)
    return x, t

def PS_Hip_test_data(seed, mu, sig, x0,  maturity, nPoints, nPaths, mu_pr = 'Nada', p0=0.5):

    np.random.seed(seed)
    if mu_pr =='Nada':
       mu_pr = mu
    x_process = lambda s, dt, e: s + mu_pr * dt + sig * np.sqrt(dt) * e
    x = GeneratePaths_onedim(spot=x0, process=x_process,
                             maturity=maturity, nPoints=nPoints, nPaths=nPaths)
    x = np.reshape(x, newshape=(nPaths, nPoints))
    dx = x[:, 1:] - x[:, :-1]
    p = np.zeros(shape=(nPaths, nPoints)) 
    p[:,0] = p0*np.ones(shape=(nPaths,))
    dt = maturity / (nPoints-1)
    for ii in range(nPoints-1):
        #print(ii)
        p[:,ii+1] = p[:,ii] + (mu/(sig**2)) * p[:,ii] * (1-p[:,ii]) * (dx[:,ii]- mu*p[:,ii]*dt )
    t = np.linspace(0, maturity, nPoints)
    return x, p, t


