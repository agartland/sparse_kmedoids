import pytest

def test_kmedoids():
    from sklearn import neighbors, datasets
    from sparse_kmedoids import kmedoids, sparse_kmedoids
    import scipy.sparse

    n_passes = 20
    k = 3
    max_iter = 1000

    iris = datasets.load_iris()
    obs = iris['data']
    dmat = neighbors.DistanceMetric.get_metric('euclidean').pairwise(obs)
    """res = medoids, labels, inertia, iters, n_found"""
    res = kmedoids(dmat, k=k, max_iter=max_iter, n_passes=n_passes)

    dmat_sparse = scipy.sparse.csr_matrix(dmat)
    res_sparse = sparse_kmedoids(dmat_sparse, k=k, max_iter=max_iter, n_passes=n_passes)
    assert np.all(res_sparse[0] == res[0])
    assert np.all(res_sparse[1] == res[1])
