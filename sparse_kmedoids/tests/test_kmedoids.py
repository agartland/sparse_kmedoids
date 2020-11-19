import pytest

def test_kmedoids(n_passes=20, k=3, max_iter=1000):
    from sklearn import neighbors, datasets
    from sparse_kmedoids import kmedoids

    iris = datasets.load_iris()
    obs = iris['data']
    dmat = neighbors.DistanceMetric.get_metric('euclidean').pairwise(obs)
    results = kmedoids(dmat, k=k, max_iter=max_iter, n_passes=n_passes)
    # return (dmat,) + results

def test_FCMdd(n_passes=20, c=3, max_iter=1000, membership_method=('FCM', 2)):
    from sklearn import neighbors, datasets
    from sparse_kmedoids import fuzzycmedoids
    iris = datasets.load_iris()
    obs = iris['data']
    dmat = neighbors.DistanceMetric.get_metric('euclidean').pairwise(obs)
    results = fuzzycmedoids(dmat, c=c, max_iter=max_iter, n_passes=n_passes, membership_method=membership_method)
    # return (dmat,) + results

@pytest.mark.skip(reason='hard to eval plotting')
def _test_plot():
    k=3
    n_passes=20
    max_iter=1000
    from sklearn import neighbors, datasets
    from Bio.Cluster import kmedoids as biokmedoids
    import time
    import matplotlib.pyplot as plt
    import palettable
    import seaborn as sns
    sns.set(style='darkgrid', palette='muted', font_scale=1.3)
    cmap = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

    iris = datasets.load_iris()
    obs = iris['data']
    dmat = neighbors.DistanceMetric.get_metric('euclidean').pairwise(obs)
    weights = np.random.rand(obs.shape[0])

    plt.figure(2)
    plt.clf()
    plt.subplot(2, 2, 1)
    startTime = time.time()
    medoids, labels, inertia, niter, nfound = kmedoids(dmat, k=k, max_iter=max_iter, n_passes=n_passes)
    et = time.time() - startTime
    for medi, med in enumerate(medoids):
        plt.scatter(obs[labels==med, 0], obs[labels==med, 1], color=cmap[medi])
        plt.plot(obs[med, 0], obs[med, 1], 'sk', markersize=10, color=cmap[medi], alpha=0.5)
    plt.title('K-medoids (%1.3f sec, %d iterations, %d solns)' % (et, niter, nfound))

    plt.subplot(2, 2, 3)
    startTime = time.time()
    medoids, labels, inertia, niter, nfound = kmedoids(dmat, k=k, max_iter=max_iter, n_passes=n_passes, weights=weights)
    et = time.time() - startTime
    for medi, med in enumerate(medoids):
        nWeights = _rangenorm(weights, mn=10, mx=200)
        plt.scatter(obs[labels==med, 0], obs[labels==med, 1], color=cmap[medi], s=nWeights, edgecolor='black', alpha=0.5)
        plt.plot(obs[med, 0], obs[med, 1], 'sk', markersize=10, color=cmap[medi])
    plt.title('Weighted K-medoids (%1.3f sec, %d iterations, %d solns)' % (et, niter, nfound))

    plt.subplot(2, 2, 2)
    startTime = time.time()
    biolabels, bioerror, bionfound = biokmedoids(dmat, nclusters=k, npass=n_passes)
    biomedoids = np.unique(biolabels)
    bioet = time.time() - startTime
    for medi, med in enumerate(biomedoids):
        plt.scatter(obs[biolabels==med, 0], obs[biolabels==med, 1], color=cmap[medi])
        plt.plot(obs[med, 0], obs[med, 1], 'sk', color=cmap[medi], markersize=10, alpha = 0.5)
    plt.title('Bio.Cluster K-medoids (%1.3f sec, %d solns)' % (bioet, bionfound))

    plt.subplot(2, 2, 4)
    startTime = time.time()
    medoids, membership, niter, nfound = fuzzycmedoids(dmat, c=k, max_iter=max_iter, n_passes=n_passes)
    labels = medoids[np.argmax(membership, axis=1)]
    et = time.time() - startTime
    
    for medi, med in enumerate(medoids):
        ind = labels == med
        sz = _rangenorm(membership[:, medi][ind], mn=10, mx=100)
        sz[np.argmax(sz)] = 0.
        plt.scatter(obs[ind, 0], obs[ind, 1], color=cmap[medi], s=sz, alpha=0.5)
        plt.plot(obs[med, 0], obs[med, 1], 'sk', markersize=10, color=cmap[medi])
    plt.title('Fuzzy c-medoids (%1.3f sec, %d iterations, %d solns)' % (et, niter, nfound))

