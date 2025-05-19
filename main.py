import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ot

import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

@jax.jit
def get_wrd(w1, w2):
    w1_norm = jnp.linalg.norm(w1, axis=1)
    m1 = w1_norm / w1_norm.sum()

    w2_norm = jnp.linalg.norm(w2, axis=1)
    m2 = w2_norm / w2_norm.sum()

    w_dot = jnp.dot(w1, w2.T)
    w_norm = jnp.outer(w1_norm, w2_norm.T)
    c = 1 - w_dot / w_norm

    geom = geometry.Geometry(cost_matrix=c)

    prob = linear_problem.LinearProblem(geom, m1, m2)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)

    return out.reg_ot_cost

class subspace_wrd:
    def __init__(self, s_vecs, k=None, learn_fn="kmeans"):
        self.s_vecs = s_vecs
        if k is None:
            self.k = max([len(w_vecs) for w_vecs in self.s_vecs])
        else:
            self.k = k
        self.learn_fn = learn_fn

    def learn_by_pca(self, all_w_vecs):
        pca = PCA(n_components=self.k)
        pca.fit_transform(all_w_vecs)

        return pca.components_

    def learn_by_kmeans(self, all_w_vecs):
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(all_w_vecs)

        return kmeans.cluster_centers_

    def learn_base_vecs(self):
        all_w_vecs = list(itertools.chain.from_iterable(self.s_vecs))
        if self.learn_fn == "kmeans":
            self.W = self.learn_by_kmeans(all_w_vecs)
        elif self.learn_fn == "pca":
            self.W = self.learn_by_pca(all_w_vecs)

    def get_fixed_s_vecs(self, X):
        a = np.ones(X.shape[0]) / X.shape[0]
        b = np.ones(self.k) / self.k

        c = ot.dist(X, self.W, metric='euclidean') ** 2

        t = ot.emd(a, b, c)

        return np.dot(t.T, X)

    def run(self):
        fixed_s_vecs = list(map(self.get_fixed_s_vecs, self.s_vecs))
        
        combinations = list(itertools.combinations(range(len(fixed_s_vecs)), 2))
        w1 = jnp.array([fixed_s_vecs[i[0]] for i in combinations])
        w2 = jnp.array([fixed_s_vecs[i[1]] for i in combinations])

        batched_get_wrd = jax.vmap(get_wrd, in_axes=(0, 0))
        self.result = batched_get_wrd(w1, w2)

if __name__ == "__main__":
    # s_vec = 
    subspace_wrd = subspace_wrd(s_vecs)
    subspace_wrd.learn_base_vecs()
    subspace_wrd.run()
    subspace_wrd.result.shape
