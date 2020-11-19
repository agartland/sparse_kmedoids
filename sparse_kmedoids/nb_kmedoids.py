import numpy as np
from .vectools import unique_rows
import scipy
import itertools
import numba

__all__ = ['sparse_kmedoids']

def sparse_kmedoids(dmat_csr, k, max_d_penalty=None, weights=None, n_passes=1, max_iter=1000, seed=110820):
    """Identify the k points that minimize all intra-cluster distances.

    The algorithm completes nPasses of the algorithm with random restarts.
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
    max_d_penalty : distance dtype
        Used to compute inertia for points beyond the max distance for the sparse distance array
    n_passes : int
        Number of times the algorithm is restarted with random medoid initializations. The best solution is returned.
    max_iter : int, optional, default None (inf)
        Maximum number of iterations of the k-medoids algorithm to run.
    seed : int or False
        If not False, sets np.random.seed for reproducible results.

    Returns
    -------
    medoids : float ndarray with shape (k)
        Indices into dmat that indicate medoids found at the last iteration of k-medoids.
    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest medoid for all observations).
    n_iter : int
        Number of iterations run.
    labels : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the medoid the
        i'th observation is closest to.
    n_found : int
        Number of unique solutions found (out of n_passes)"""
    if weights is None:
        weights = np.ones(dmat_csr.shape[0], dtype=dmat_csr.dtype)
    if max_d_penalty is None:
        max_d_penalty = np.max(dmat_csr.data) + 1

    medoids, inertias, iters, best_pass, best_labels  = kmedoids_passes(dmat_csr.data,
                                                            dmat_csr.indices,
                                                            dmat_csr.indptr,
                                                            k=k,
                                                            max_d_penalty=max_d_penalty,
                                                            weights=weights,
                                                            n_passes=n_passes,
                                                            max_iter=max_iter,
                                                            seed=seed)
    
    """n_found is the number of unique solutions (each row is a solution)"""
    n_found = unique_rows(medoids).shape[0]
    return sorted(medoids[best_pass, :]), best_labels, inertias[best_pass], iters[best_pass], n_found

@numba.jit(nopython=True, parallel=False, nogil=True)
def kmedoids_passes(data, indices, indptr, k, max_d_penalty, weights, n_passes, max_iter, seed):
    """Use a seed to guarantee reproducibility"""
    np.random.seed(seed)

    """Number of original points"""
    n = indptr.shape[0] - 1

    best_labels = np.zeros(n, dtype=numba.int_)
    best_inertia = -1
    best_pass = -1
    inertias = np.zeros(n_passes, dtype=data.dtype)
    medoids = np.zeros((n_passes, k), dtype=numba.int_)
    iters = np.zeros(n_passes, dtype=numba.int_)
    for passi in range(n_passes):
        curr_medoids, labels, tot_inertia, i = inner_kmedoids(data, indices, indptr, k, weights, max_d_penalty, max_iter)
        if tot_inertia < best_inertia or best_inertia == -1:
            best_inertia = tot_inertia
            best_labels = labels
            best_pass = passi
        medoids[passi, :] = curr_medoids
        inertias[passi] = tot_inertia
        iters[passi] = i

    """Return the results from the best pass"""
    return medoids, inertias, iters, best_pass, best_labels

def _get_csr_row(csr_data, indptr, rowi):
    return csr_data[indptr[rowi]:indptr[rowi + 1]]

@numba.jit(nopython=True, parallel=False, nogil=True)
def inner_kmedoids(data, indices, indptr, k, weights, max_d_penalty, max_iter):
    """Pick k random medoids"""
    n = indptr.shape[0] - 1
    curr_medoids = np.random.permutation(n)[:k]
    new_medoids = np.zeros(k, dtype=numba.int_)
    labels = curr_medoids[np.random.choice(np.arange(k), n)]
    curr_dist = (np.max(data) + 1) * np.ones(n, dtype=data.dtype)
    for i in range(max_iter):
        """Assign each point to the closest cluster,
        but don't reassign a point if the distance isn't an improvement."""
        assign_clusters(data, indices, indptr, curr_medoids, curr_dist, labels)
        
        """If clusters are lost during (re)assignment step, pick random points
        as new medoids and reassign until we have k clusters again"""
        ulabels = np.unique(labels)
        while ulabels.shape[0] < k:
            for medi, curr_med in enumerate(curr_medoids):
                if ~np.any(curr_med == ulabels):
                    choice = np.random.randint(n)
                    while np.any(choice == ulabels):
                        choice = np.random.randint(n)
                    curr_medoids[medi] = choice
                    assign_clusters(data, indices, indptr, curr_medoids, curr_dist, labels)
                    ulabels = np.unique(labels)
                    break

        """ISSUE: If len(unique(labels)) < k there is an error"""

        """Choose new medoids for each cluster, minimizing intra-cluster distance"""
        tot_inertia = 0
        for medi, curr_med in enumerate(curr_medoids):
            cluster_ind = np.nonzero(labels == curr_med)[0]
            """Inertia is the sum of the distances (vec is shape (len(clusterInd))"""
            """Look at all the potential new medoids within this cluster, find the inertia
            to the current medoid and the new medoids and assign the new medoid to minimize inertia"""
            inertia_vec = np.zeros(cluster_ind.shape[0], dtype=data.dtype)
            min_inertia = -1
            min_med = -1
            for new_med in cluster_ind:
                """Go through each potential new medoid within this cluster and compute inertia"""
                inertia = 0
                for rowi in cluster_ind:
                    """Go through each cluster member and compute inertia to the candidate medoid"""
                    row = data[indptr[rowi]:indptr[rowi + 1]]
                    col_indices = indices[indptr[rowi]:indptr[rowi + 1]]
                    addendum = max_d_penalty
                    for cii, col_index in enumerate(col_indices):
                        if new_med == col_index:
                            if row[cii] > 0:
                                """Assume that negative distances are actually zeros and missing values are max_d"""
                                addendum = row[cii] * weights[rowi]
                            break
                    inertia += addendum
                if min_inertia == -1 or inertia < min_inertia:
                    """Keep track of medoid with for this cluster min inertia"""
                    min_inertia = inertia
                    min_med = new_med
                # if new_med == med:
                    """Keep inertia to current medoid"""
                    # curr_inertia = inertia
            # if min_inertia < curr_inertia:
            """If the new best medoid is different than the current one, update!
            (just update because the curr medoid was in there too"""
            new_medoids[medi] = min_med

            """Add inertia of this new medoid to the running total"""
            tot_inertia += min_inertia

        if np.all(new_medoids == curr_medoids):
            """If the medoids didn't need to be updated then we're done!"""
            break
        curr_medoids = new_medoids.copy()
    return curr_medoids, labels, tot_inertia, i

@numba.jit(nopython=True, parallel=False, nogil=True)
def assign_clusters(data, indices, indptr, curr_medoids, curr_dist, labels):
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

    n = indptr.shape[0] - 1
    k = curr_medoids.shape[0]

    """Assign each point to the closest cluster,
    but don't reassign a point if the distance isn't an improvement."""
    for rowi in np.arange(n):
        """With each data point, get the distances to all nearby points"""
        row = data[indptr[rowi]:indptr[rowi + 1]]
        if len(row) > 0:
            """If there are any nearby points"""
            if ~np.any(labels[rowi] == curr_medoids):
                curr_dist[rowi] = -1
            col_indices = indices[indptr[rowi]:indptr[rowi + 1]]
            for cii, col_index in enumerate(col_indices):
                if np.any(col_index == curr_medoids):
                    """If its a nearby point and a medoid check the distance"""
                    if row[cii] < 0:
                        """Assume that negative distances are actually zeros and missing values are max_d"""
                        d = 0
                    else:
                        d = row[cii]
                    if d < curr_dist[rowi] or curr_dist[rowi] == -1:
                        labels[rowi] = col_index
                        curr_dist[rowi] = d
        else:
            labels[rowi] = -1

    return