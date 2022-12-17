import faiss
import numpy as np


class Kmeans(object):
    def __init__(self, nmb_clusters, nredo=20):
        """
        Kmeans clustering.

        Args:
        nmb_clusters (int): The numbers of clusters.
        nredo (int): The number os trials in clustering.
        """
        self.nmb_clusters = nmb_clusters
        self.nredo = nredo

    def cluster(self, npdata1, npdata2=None, use_pca=False, save_centroids=False):

        # PCA-reducing, whitening and L2-normalization
        npdata1, npdata2 = self.preprocess_features(npdata1, npdata2, use_pca=use_pca)

        # Clustering the data
        labels, loss, centroids = self.run_kmeans(
            npdata1=npdata1,
            npdata2=npdata2,
            nmb_clusters=self.nmb_clusters,
            nredo=self.nredo,
            save_centroids=save_centroids,
        )
        if save_centroids:
            self.centroids = faiss.vector_to_array(centroids)

        self.labels = labels
        return loss

    def preprocess_features(self, npdata1, npdata2, pca_dim=128, use_pca=False):
        _, ndim = npdata1.shape
        npdata1 = npdata1.astype("float32")

        # Apply PCA-whitening with Faiss
        if use_pca:
            mat = faiss.PCAMatrix(ndim, pca_dim, eigen_power=-0.5)
            mat.train(npdata1)
            assert mat.is_trained
            npdata1 = mat.apply_py(npdata1)
            if npdata2 is not None:
                npdata2 = mat.apply_py(npdata2)

        # L2 normalization
        row_sums = np.linalg.norm(npdata1, axis=1)
        npdata1 = npdata1 / (row_sums[:, np.newaxis] + 1e-8)
        if npdata2 is not None:
            row2_sums = np.linalg.norm(npdata2, axis=1)
            npdata2 = npdata2 / (row2_sums[:, np.newaxis] + 1e-8)

        return npdata1, npdata2
    
    def run_kmeans(
        self,
        npdata1,
        npdata2=None,
        nmb_clusters=50,
        nredo=20,
        save_centroids=False,
    ):
        d = npdata1.shape[-1]

        # faiss implementation of k-means
        clus = faiss.Clustering(d, nmb_clusters)

        # Change faiss seed at each k-means so that the randomly picked
        # initialization centroids do not correspond to the same feature ids
        # from an epoch to another.
        clus.seed = np.random.randint(1234)

        clus.nredo = nredo
        clus.niter = 20
        clus.max_points_per_centroid = 10000000
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = False
        flat_config.device = 0
        index = faiss.GpuIndexFlatL2(res, d, flat_config)
        # perform the training
        clus.train(npdata1, index)

        # find the classes
        if npdata2 is not None:
            npdata1 = np.concatenate((npdata1, npdata2), axis=0)
        D, I = index.search(npdata1, 1)

        stats = clus.iteration_stats
        losses = np.array([stats.at(i).obj for i in range(stats.size())])

        if save_centroids:
            centroids = clus.centroids
            return [int(n[0]) for n in I], losses[-1], centroids
        else:
            return [int(n[0]) for n in I], losses[-1], None
