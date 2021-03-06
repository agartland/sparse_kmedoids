import numpy as np
import scipy
import itertools
import warnings

from .vectools import unique_rows

__all__ = ['kmedoids',
           'fuzzycmedoids',
           'tryallmedoids',
           'precomputeWeightedDmat',
           'assignClusters',
           'computeInertia',
           'computeMembership']

def kmedoids(dmat, k=3, weights=None, n_passes=1, max_iter=1000, init_inds=None, potential_medoid_inds=None, seed=110820):
    """Identify the k points that minimize all intra-cluster distances.

    The algorithm completes n_passes of the algorithm with random restarts.
    Each pass consists of iteratively assigning/improving the medoids.
    
    Uses Partioning Around Medoids (PAM) as the EM.

    To apply to points in euclidean space pass dmat using:
    dmat = sklearn.neighbors.DistanceMetric.get_metric('euclidean').pairwise(points_array)
    
    Parameters
    ----------
    dmat : array-like of floats, shape (n_samples, n_samples)
        The pairwise distance matrix of observations to cluster.
    weights : array-like of floats, shape (n_samples)
        Relative weights for each observation in inertia computation.
    k : int
        The number of clusters to form as well as the number of
        medoids to generate.
    n_passes : int
        Number of times the algorithm is restarted with random medoid initializations. The best solution is returned.
    max_iter : int, optional, default None (inf)
        Maximum number of iterations of the k-medoids algorithm to run.
    init_inds : ndarray
        Medoid indices used for random initialization and restarts for each pass.
    potential_medoid_inds : array of indices
        If specified then medoids are constrained to be chosen from this array.
    seed : int or False
        If not False, sets np.random.seed for reproducible results.

    Returns
    -------
    medoids : float ndarray with shape (k)
        Indices into dmat that indicate medoids found at the last iteration of k-medoids.
    labels : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the medoid the
        i'th observation is closest to.
    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest medoid for all observations).
    nIter : int
        Number of iterations run.
    nFound : int
        Number of unique solutions found (out of n_passes)"""

    """Use a seed to guarantee reproducibility"""
    if seed:
        np.random.seed(seed)

    """Number of points"""
    N = dmat.shape[0]

    if init_inds is None:
        init_inds = np.arange(N)

    wdmat = precomputeWeightedDmat(dmat, weights, squared=True)

    if not potential_medoid_inds is None:
        potentialMedoidSet = set(potential_medoid_inds)
        init_inds = np.array([i for i in potentialMedoidSet.intersection(set(init_inds))], dtype=int)
    else:
        potentialMedoidSet = set(np.arange(N))

    if len(init_inds)==0:
        print('No possible init_inds provided.')
        return

    bestInertia = None
    allMedoids = np.zeros((n_passes, k))
    for passi in range(n_passes):
        """Pick k random medoids"""
        currMedoids = np.random.permutation(init_inds)[:k]
        newMedoids = np.zeros(k, dtype=int)
        labels = currMedoids[np.random.randint(k, size=N)]
        for i in range(max_iter):
            """Assign each point to the closest cluster,
            but don't reassign a point if the distance isn't an improvement."""
            labels = assignClusters(dmat, currMedoids, oldLabels=labels)
            
            """If clusters are lost during (re)assignment step, pick random points
            as new medoids and reassign until we have k clusters again"""
            uLabels = np.unique(labels[potential_medoid_inds])
            while uLabels.shape[0]<k:
                for medi, med in enumerate(currMedoids):
                    if not med in uLabels:
                        choices = list(set(init_inds).difference(set(uLabels)))
                        currMedoids[medi] = choices[np.random.randint(len(choices))]
                        
                        labels = assignClusters(dmat, currMedoids, oldLabels=labels)
                        uLabels = np.unique(labels[potential_medoid_inds])
                        break

            """ISSUE: If len(unique(labels)) < k there is an error"""

            """Choose new medoids for each cluster, minimizing intra-cluster distance"""
            totInertia = 0
            for medi, med in enumerate(currMedoids):
                clusterInd = np.where(labels == med)[0]
                """Limit medoids to those specified by indexing axis=0 with the intersection of potential medoids and all points in the cluster"""
                potentialInds = np.array([poti for poti in potentialMedoidSet.intersection(set(clusterInd))])
                """Inertia is the sum of the distances (vec is shape (len(clusterInd))"""
                inertiaVec = (wdmat[potentialInds,:][:, clusterInd]).sum(axis=1)
                mnInd = np.argmin(inertiaVec)
                newMedoids[medi] = potentialInds[mnInd]
                """Add inertia of this new medoid to the running total"""
                totInertia += inertiaVec[mnInd]

            if (newMedoids == currMedoids).all():
                """If the medoids didn't need to be updated then we're done!"""
                allMedoids[passi,:] = sorted(currMedoids)
                break
            currMedoids = newMedoids.copy()
        if bestInertia is None or totInertia < bestInertia:
            """For multiple passes, see if this pass was better than the others"""
            bestInertia = totInertia
            bestMedoids = currMedoids.copy()
            bestLabels = labels.copy()
            bestNIter = i + 1
    
    """nfound is the number of unique solutions (each row is a solution)"""
    nfound = unique_rows(allMedoids).shape[0]
    """Return the results from the best pass"""
    return bestMedoids, bestLabels, bestInertia, bestNIter, nfound

def precomputeWeightedDmat(dmat, weights, squared=False):
    """Compute the weighted distance matrix for kmedoids and FCMdd.
   
    Adding weight to a point increases its impact on inertia linearly,
    such that the algorithm will tend to favor minimization of distances
    to that point.

    Note: weights are applied along the rows so to correctly compute
    inertia for a given medoid one would index the cluster along axis=1,
    index the medoid along axis=0 and then sum along axis=1.

    Parameters
    ----------
    dmat : ndarray shape[N x N]
        Pairwise distance matrix (unweighted).
    weights : ndarray shape[N]
        Should sum to one if one wants to preserve
        inertial units with different weights.

    Returns
    -------
    wdmat : ndarray shape[N x N]
        Weighted distance matrix, ready for computing inertia."""
    
    if squared:
        dmat = dmat**2

    if weights is None:
        return dmat
    else:
        assert weights.shape[0] == dmat.shape[0]
        return dmat * weights[None,:].values

def assignClusters(dmat, currMedoids, oldLabels=None):
    """Assigns/reassigns points to clusters based on the minimum (unweighted) distance.
    
    Note: if oldLabels are specified then only reassigns points that
    are not currently part of a cluster that minimizes their distance.
    
    This ensures that when there are ties for best cluster with the current cluster,
    the point is not reassigned to a new cluster.

    Parameters
    ----------
    dmat : ndarray shape[N x N]
        Pairwise distance matrix (unweighted).
    currMedoids : ndarray shape[k]
        Index into points/dmat that specifies the k current medoids.
    oldLabels : ndarray shape[N]
        Old labels that will be reassigned.

    Returns
    -------
    labels : ndarray shape[N]
        New labels such that unique(labels) equals currMedoids."""

    N = dmat.shape[0]
    k = len(currMedoids)

    """Assign each point to the closest cluster,
    but don't reassign a point if the distance isn't an improvement."""
    if not oldLabels is None:
        labels = oldLabels
        oldD = dmat[np.arange(N), labels]
        minD = (dmat[:, currMedoids]).min(axis=1)
        """Points where reassigning is neccessary"""
        reassignInds = (minD<oldD) | ~np.any(np.tile(labels[:, None], (1, k)) == np.tile(currMedoids[None,:], (N, 1)), axis=1)
    else:
        reassignInds = np.arange(N)
        labels = np.zeros(N)
    labels[reassignInds] = currMedoids[np.argmin(dmat[reassignInds,:][:, currMedoids], axis=1)]
    return labels

def computeInertia(wdmat2, labels, currMedoids):
    """Computes inertia for a set of clustered points using
    a precomputed weighted distance matrix.
    
    Note: wdmat needs to be summed along axis=1

    assert all(sorted(unique(labels)) == sorted(currMedoids))

    Parameters
    ----------
    wdmat : ndarray shape[N x N]
        Weighted and squared distance matrix, ready for computing inertia.
    labels : ndarray shape[N]
        Cluster assignment (medoid index) of each point
    currMedoids : ndarray shape[k]
        Index into points/dmat that specifies the k current medoids.

    Returns
    -------
    inertia : float
        Total inertia of all k clusters"""

    assert np.all(np.unique(labels) == np.unique(currMedoids))
    
    totInertia = 0
    for medi, med in enumerate(currMedoids):
        clusterInd = np.where(labels == med)[0]
        """Inertia is the sum of the distances"""
        totInertia += wdmat2[med, clusterInd].sum()
    return totInertia

def fuzzycmedoids(dmat, c, membership_method=('FCM', 2), weights=None, n_passes=1, max_iter=1000, init_inds=None, potential_medoid_inds=None):
    """Implementation of fuzz c-medoids (FCMdd)

    Krishnapuram, Raghu, Anupam Joshi, Liyu Yi, Computer Sciences, and Baltimore County.
        "A Fuzzy Relative of the K-Medoids Algorithm with Application to Web Documents."
        Electrical Engineering. doi:10.1109/FUZZY.1999.790086.

    The algorithm completes n_passes of the algorithm with random restarts.
    Each pass consists of iteratively assigning/improving the medoids.
    
    To apply to points in euclidean space pass dmat using:
    dmat = sklearn.neighbors.DistanceMetric.get_metric('euclidean').pairwise(points_array)
    
    Parameters
    ----------
    dmat : array-like of floats, shape (n_samples, n_samples)
        Pairwise distance matrix of observations to cluster.
    weights : array-like of floats, shape (n_samples)
        Relative weights for each observation in inertia computation.
    c : int
        The number of clusters to form as well as the number of medoids to generate.
    membership_method : tuple of (method str/int, param)
        Method for computing membership matrix.
    n_passes : int
        Number of times the algorithm is restarted with random medoid initializations.
        The best solution is returned.
    max_iter : int, optional, default None (inf)
        Maximum number of iterations of the c-medoids algorithm to run.
    init_inds : ndarray
        Medoid indices used for random initialization and restarts for each pass.
    potential_medoid_inds : array of indices
        If specified, then medoids are constrained to be chosen from this array.

    Returns
    -------
    medoids : float ndarray with shape (c)
        Indices into dmat that indicate medoids found at the last iteration of FCMdd
    membership : float ndarray with shape (n_samples, c)
        Each row contains the membership of a point to each of the clusters.
    nIter : int
        Number of iterations run.
    nFound : int
        Number of unique solutions found (out of n_passes)"""

    """Number of points"""
    N = dmat.shape[0]

    if init_inds is None:
        init_inds = np.arange(N)

    wdmat = precomputeWeightedDmat(dmat, weights)

    if not potential_medoid_inds is None:
        init_inds = np.array([i for i in init_inds if i in potential_medoid_inds], dtype=int)
    else:
        potential_medoid_inds = np.arange(N)

    if len(init_inds) == 0:
        print('No possible init_inds provided.')
        return

    allMedoids = np.zeros((n_passes, c))
    bestInertia = None
    foundSame = 0
    for passi in range(n_passes):
        """Pick c random medoids"""
        currMedoids = np.random.permutation(init_inds)[:c]
        newMedoids = np.zeros(c, dtype=int)
        for i in range(max_iter):
            """(Re)compute memberships [N x c]"""
            membership = computeMembership(dmat, currMedoids, method=membership_method[0], param=membership_method[1])
            
            """Choose new medoid for each cluster, minimizing fuzzy objective function"""
            totInertia = 0
            for medi, med in enumerate(currMedoids):
                """Within each cluster find the new medoid
                by minimizing the dissimilarities,
                weighted by membership to the cluster"""

                """Inertia is the sum of the membership times the distance matrix over all points.
                (membership for cluster medi [a column vector] is applied across the columns of wdmat
                [and broadcast to all row vectors] before summing)"""
                inertiaMat = np.tile(membership[:, medi][:, None].T, (len(potential_medoid_inds), 1)) * wdmat[potential_medoid_inds,:]
                inertiaVec = inertiaMat.sum(axis=1)
                mnInd = np.argmin(inertiaVec)
                newMedoids[medi] = potential_medoid_inds[mnInd]
                """Add inertia of this new medoid to the running total"""
                totInertia += inertiaVec[mnInd]

            if (newMedoids == currMedoids).all():
                """If the medoids didn't need to be updated then we're done!"""
                allMedoids[passi,:] = sorted(currMedoids)
                break
            currMedoids = newMedoids.copy()
            
        if bestInertia is None or totInertia < bestInertia:
            """For multiple passes, see if this pass was better than the others"""
            bestInertia = totInertia
            bestMedoids = currMedoids.copy()
            bestMembership = membership.copy()
            bestNIter = i + 1
    
    """nfound is the number of unique solutions (each row is a solution)"""
    nfound = unique_rows(allMedoids).shape[0]
    """Return the results from the best pass"""
    return bestMedoids, bestMembership, bestNIter, nfound#, allMedoids

def fuzzyPartitionCoef(membership):
    """Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix membership. Measures 'fuzziness' in partitioned clustering.

    Copied from from sckit-fuzzy:
    https://github.com/scikit-fuzzy/scikit-fuzzy
    
    Parameters
    ----------
    membership : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient."""
    n = membership.shape[1]

    return np.trace(membership.dot(membership.T)) / float(n)

def computeMembership(dmat, medoids, method='FCM', param=2):
    """Compute membership of each instance in each cluster,
    defined by the provided medoids.

    Possible methods come from the manuscript by Krishnapuram et al.
    and may include an additional parameter (typically a "fuzzifier")

    Parameters
    ----------
    dmat : ndarray shape[N x N]
        Pairwise distance matrix (unweighted).
    medoids : ndarray shape[c]
        Index into points/dmat that specifies the c current medoids.
    method : str
        Method for computing memberships:
            "FCM" (from fuzzy c-means)
            Equations from Krishnapuram et al. (2, 3, 4 or 5)
    param : float
        Additional parameter required by the method.
        Note: param must be shape[c,] for methods 4 or 5

    Returns
    -------
    membership : ndarray shape[N x c]
        New labels such that unique(labels) equals currMedoids."""

    N = dmat.shape[0]
    c = len(medoids)

    r = dmat[:, medoids]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method in ['FCM', 2, '2']:
            assert param >= 1
            tmp = (1 / r)**(1 / (param - 1))
        elif method in [3, '3']:
            assert param > 0
            tmp = np.exp(-param * r)
        elif method in [4, '4']:
            assert param.shape == (c,)
            tmp = 1/(1 + r/param[:, None])
        elif method in [5, '5']:
            assert param.shape == (c,)
            tmp = np.exp(-r/param[:, None])

        membership = tmp / tmp.sum(axis=1, keepdims=True)
    for medi, med in enumerate(medoids):
        membership[med,:] = 0.
        membership[med, medi] = 1.
    return membership

def tryallmedoids(dmat, c, weights=None, potential_medoid_inds=None, fuzzy=True, fuzzyParams=('FCM', 2)):
    """Brute force optimization of k-medoids or fuzzy c-medoids clustering.

    To apply to points in euclidean space pass dmat using:
    dmat = sklearn.neighbors.DistanceMetric.get_metric('euclidean').pairwise(points_array)
    
    Parameters
    ----------
    dmat : array-like of floats, shape (n_samples, n_samples)
        Pairwise distance matrix of observations to cluster.
    c : int
        Number of clusters to form as well as the number of medoids to generate.
    weights : array-like of floats, shape (n_samples)
        Relative weights for each observation in inertia computation.
    potential_medoid_inds : array of indices
        If specified, then medoids are constrained to be chosen from this array.
    fuzzy : boolean
        If True, use fuzzy inertia function,
        otherwis use crisp cluster definition.
    fuzzyParams : tuple of (method str/int, param)
        Method and parameter for computing fuzzy membership matrix.
    
    Returns
    -------
    medoids : float ndarray with shape (c)
        Indices into dmat that indicate optimal medoids.
    membership or labels: float ndarray with shape (n_samples, c) or shape (n_samples,)
        Each row contains the membership of a point to each of the clusters
        OR with hard clusters, the medoid/cluster index of each point."""

    if fuzzy:
        wdmat = precomputeWeightedDmat(dmat, weights, squared=False)
    else:
        wdmat = precomputeWeightedDmat(dmat, weights, squared=True)

    N = dmat.shape[0]

    if potential_medoid_inds is None:
        potential_medoid_inds = np.arange(N)

    combinations = scipy.misc.comb(len(potential_medoid_inds), c)
    if combinations > 1e7:
        print("Too many combinations to try: %1.1g > 10M" % combinations)

    bestInertia = None
    for medInds in itertools.combinations(list(range(len(potential_medoid_inds))), c):
        medoids = potential_medoid_inds[np.array(medInds)]

        if fuzzy:
            membership = computeMembership(dmat, medoids, method=fuzzyParams[0], param=fuzzyParams[1])
        else:
            membership = np.zeros((N, c))
            membership[np.arange(N), np.argmin(dmat[:, medoids], axis=1)] = 1.
        inertia = (wdmat[:, medoids] * membership).sum()

        if bestInertia is None or inertia < bestInertia:
            bestMedoids = medoids
            bestInertia = inertia
            bestMembership = membership

    if not fuzzy:
        membership = np.argmax(membership, axis=1)
    return medoids, membership

def _rangenorm(vec, mx=1, mn=0):
    """Normazlize values of vec in-place to [mn, mx] interval"""
    vec = vec - np.nanmin(vec)
    vec = vec / np.nanmax(vec)
    vec = vec * (mx-mn) + mn
    return vec