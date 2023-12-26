""" This module contains the implementation of the Synchronization Likelihood Algorithm."""

import os
import numpy as np
import scipy as sp
from scipy.io import savemat, loadmat
from numba import njit


class Synchronization(object):
    """
    Class for the Synchronization Likelihood Algorithm.
    Parameters
    ----------
    data : ndarray
        Time series.
    m : int
        Embedding dimension.
    lag : int
        Lag.
    w1 : int
        Lower bound of the window.
    w2 : int
        Upper bound of the window.
    pref : float
        Preferred probability.

    Attributes
    ----------
    m : int
        Embedding dimension.
    lag : int
        Lag.
    w1 : int
        Lower bound of the window.
    w2 : int
        Upper bound of the window.
    pref : float
        Preferred probability.
    E : dict
        Critical epsilon for each time series.
    D : dict
        Distances between all pairs of points in the embedding of each time series.
    X : dict
        Embedding of each time series.
    data : ndarray
        Time series.

    Methods
    -------
    _get_embeddings(x, m, lag) : ndarray
        Embedding of a time series x with embedding dimension m and lag l.
    _get_distances(X) : ndarray
        Compute the distances between all pairs of points in X.
    _get_prob(D, eps, i, w1, w2) : float
        Compute the probability of a point i being in the epsilon-neighborhood of another point j.
    _obj(eps, D, i, w1, w2, pref) : float
        Objective function for the root finding algorithm.
    _get_critical_eps(D, pref, w1, w2, brack) : ndarray
        Compute the critical epsilon for a given preferred probability.
    _get_Hij(D1, D2, E1, E2, w1, w2) : ndarray
        Compute the H matrix.
    _calc_SL(D, E, H) : ndarray
        Compute the synchronization likelihood matrix.
    _avg(SL, w1, w2) : ndarray
        Compute the average synchronization likelihood.
    SyncLike(x, y) : ndarray
        Compute the synchronization likelihood for two time series.
    it(size) : generator
        Generator for the synchronization likelihood of all pairs of time series.
    synchronization() : ndarray
        Compute the synchronization likelihood for all pairs of time series.


    Examples
    --------
    >>> import numpy as np
    >>> from synclike import Synchronization
    >>> data = np.random.random((5, 2500))
    >>> SL = Synchronization(data, 5, 2, 100, 410, 0.05)
    >>> SL.synchronization()

    References
    ----------
    .. [1] Stam, C.J., and B.W. Van Dijk. "Synchronization Likelihood: An Unbiased Measure of Generalized Synchronization in Multivariate Data Sets." Physica D: Nonlinear Phenomena, vol. 163, no. 3-4, 2002, pp. 236-251,  https://doi.org/10.1016/S0167-2789(01)00386-4. Accessed 19 Dec. 2023.
    """

    def __init__(self, data, m, lag, w1, w2, pref) -> None:
        self.m, self.lag, self.w1, self.w2, self.pref = m, lag, w1, w2, pref
        self.E = dict()
        self.D = dict()
        self.X = dict()
        self.data = data

    @staticmethod
    @njit
    def _get_embeddings(x, m, lag) -> np.ndarray:
        """Embedding of a time series x with embedding dimension m and lag l.
        Parameters
        ----------
        x : ndarray
            Time series.
        m : int
            Embedding dimension.
        lag : int
            Lag.

        Returns
        -------
        X : ndarray
            Embedding of x.
        """
        size = len(x) - (m - 1) * lag
        X = np.zeros((m, size), dtype=np.float32)
        for i in range(m):
            X[i] = x[i * lag: i * lag + size]

        return X.T

    @staticmethod
    @njit
    def _get_distances(X) -> np.ndarray:
        """Compute the distances between all pairs of points in X.
        Parameters
        ----------
        X : ndarray
            Embedding of a time series.

        Returns
        -------
        D : ndarray
            Distances between all pairs of points in X.
        """
        t = len(X)
        D = np.zeros((t, t), dtype=np.float32)
        for i in range(t):
            for j in range(i):
                D[j, i] = D[i, j] = np.linalg.norm(X[i] - X[j])

        return D

    @staticmethod
    @njit
    def _get_prob(D, eps, i, w1, w2) -> float:
        """Compute the probability of a point i being in the
            epsilon-neighborhood of another point j.
        Parameters
        ----------
        D : ndarray
            Distances between all pairs of points in X.
        eps : float
            Epsilon.
        i : int
            Index of the point.
        w1 : int
            Lower bound of the window.
        w2 : int
            Upper bound of the window.

        Returns
        -------
        prob : float
            Probability of a point i being in the
            epsilon-neighborhood of another point j.
        """
        summ = 0
        for j in range(len(D)):
            if (w1 < abs(i - j) < w2 and D[i, j] < eps):
                summ += 1
        return summ / (2 * (w2 - w1))

    @staticmethod
    def _obj(eps, D, i, w1, w2, pref) -> float:
        """Objective function for the root finding algorithm.
        Parameters
        ----------
        eps : float
            Epsilon.
        D : ndarray
            Distances between all pairs of points in X.
        i : int
            Index of the point.
        w1 : int
            Lower bound of the window.
        w2 : int
            Upper bound of the window.
        pref : float
            Preferred probability.

        Returns
        -------
        obj : float
            Objective function.
        """
        return Synchronization._get_prob(D, eps, i, w1, w2) - pref

    @staticmethod
    def _get_critical_eps(D, pref, w1, w2, brack) -> np.ndarray:
        """Compute the critical epsilon for a given preferred probability.
        Parameters
        ----------
        D : ndarray
            Distances between all pairs of points in X.
        pref : float
            Preferred probability.
        w1 : int
            Lower bound of the window.
        w2 : int
            Upper bound of the window.
        brack : list
            Bracketing interval for the root finding algorithm.

        Returns
        -------
        e : ndarray
            Critical epsilon.
        """
        e = np.zeros(len(D), dtype=np.float32)

        for i in range(len(D)):
            e[i] = sp.optimize.root_scalar(Synchronization._obj, args=(
                D, i, w1, w2, pref), bracket=brack, method='brentq').root

        return e

    @staticmethod
    @njit
    def _get_Hij(D1, D2, E1, E2, w1, w2) -> np.ndarray:
        """Compute the H matrix.
        Parameters
        ----------
        D1 : ndarray
            Distances between all pairs of points in X1.
        D2 : ndarray
            Distances between all pairs of points in X2.
        E1 : ndarray
            Critical epsilon for X1.
        E2 : ndarray
            Critical epsilon for X2.

        Returns
        -------
        hij : ndarray
            H matrix.
        """
        hij = np.zeros((len(D1), len(D2)), dtype=np.int16)

        for i in range(len(D1)):
            for j in range(len(D2)):
                if (w1 < abs(i - j) < w2):
                    hij[i, j] += ((D1[i, j] < E1[i]) + (D2[i, j] < E2[i]))

        return hij

    @staticmethod
    @njit
    def _calc_SL(D, E, H) -> np.ndarray:
        """Compute the synchronization likelihood matrix.
        Parameters
        ----------
        D : ndarray
            Distances between all pairs of points in X.
        E : ndarray
            Critical epsilon.
        H : ndarray
            H matrix.

        Returns
        -------
        Sij : ndarray
            Synchronization likelihood matrix.
        """
        Sij = np.zeros((len(D), len(D)), dtype=np.float32)

        for i in range(len(D)):
            for j in range(len(D)):
                Sij[i, j] = H[i, j] - 1 if (D[i, j] < E[i]) else 0

        return Sij

    @staticmethod
    @njit
    def _avg(sl, w1, w2) -> np.ndarray:
        """Compute the average synchronization likelihood.
        Parameters
        ----------
        sl : ndarray
            Synchronization likelihood matrix.
        w1 : int
            Lower bound of the window.
        w2 : int
            Upper bound of the window.

        Returns
        -------
        S : ndarray
            Average synchronization likelihood.
        """
        S = np.zeros(len(sl), dtype=np.float32)

        for i in range(len(sl)):
            summ = 0
            for j in range(len(sl)):
                if (w1 < abs(i - j) < w2):
                    summ += sl[i, j]
            S[i] = summ / (2 * (w2 - w1))

        return S

    def SyncLike(self, x, y) -> np.ndarray:
        """Compute the synchronization likelihood for two time series.
        Parameters
        ----------
        x : int
            Index of the first time series.
        y : int
            Index of the second time series.

        Returns
        -------
        SL1 : ndarray
            Average synchronization likelihood for x.
        SL2 : ndarray
            Average synchronization likelihood for y.
        """

        m, lag, w1, w2, pref = self.m, self.lag, self.w1, self.w2, self.pref

        if (x not in self.X):
            X1 = Synchronization._get_embeddings(self.data[x], m, lag)
            D1 = Synchronization._get_distances(X1)
            E1 = Synchronization._get_critical_eps(
                D1, pref, w1, w2, [D1.min(), D1.max()])
            self.X[x] = X1
            self.D[x] = D1
            self.E[x] = E1
        else:
            X1 = self.X[x]
            D1 = self.D[x]
            E1 = self.E[x]

        if (y not in self.X):
            X2 = Synchronization._get_embeddings(self.data[y], m, lag)
            D2 = Synchronization._get_distances(X2)
            E2 = Synchronization._get_critical_eps(
                D2, pref, w1, w2, [D2.min(), D2.max()])
            self.X[y] = X2
            self.D[y] = D2
            self.E[y] = E2
        else:
            X2 = self.X[y]
            D2 = self.D[y]
            E2 = self.E[y]

        H = Synchronization._get_Hij(D1, D2, E1, E2, self.w1, self.w2)

        SLij1 = Synchronization._calc_SL(D1, E1, H)
        SLij2 = Synchronization._calc_SL(D2, E2, H)

        SL1 = Synchronization._avg(SLij1, w1, w2)
        SL2 = Synchronization._avg(SLij2, w1, w2)

        return SL1/0.05, SL2/0.05

    @staticmethod
    @njit
    def it(size) -> tuple:
        """Generator for the synchronization likelihood of all pairs of time series.
        Parameters
        ----------
        size : int
            Number of time series.

        Yields
        -------
        x : int
            Index of the first time series.
        y : int
            Index of the second time series.
        """

        for x in range(size):
            for y in range(x + 1, size):
                yield x, y

    def synchronization(self) -> None:
        """Compute the synchronization likelihood for all pairs of time series.

        Returns
        -------
        out : ndarray
            Synchronization likelihood matrix.
        """

        size = self.data.shape[0]
        # t = self.data.shape[1]
        # out = [[None for i in range(size)] for j in range(size)]

        if (not os.path.exists("./sync")):
            os.mkdir("sync")

        os.chdir("sync")

        for i, j in Synchronization.it(size):
            if (os.path.exists(f"sync_{i}_{j}.mat") and os.path.exists(f"sync_{j}_{i}.mat")):
                continue

            print(f"SAVING {i} {j}")
            SL1, SL2 = Synchronization.SyncLike(self, i, j)
            savemat(f"sync_{i}_{j}.mat", {"data": SL1})
            savemat(f"sync_{j}_{i}.mat", {"data": SL2})

        os.chdir("..")

        # for i in range(size):
        #     out[i][i] = np.ones_like(out[0][1])
        # out = np.array(out, dtype=np.float16)

    def connectivity(self) -> np.ndarray:
        """Compute the connectivity matrix.

        Returns
        -------
        out : ndarray
            Connectivity matrix.
        """
        size = self.data.shape[0]

        out = [None] * size

        if (not os.path.exists("./sync")):
            os.mkdir("sync")
            self.synchronization()

        os.chdir("sync")

        for i, j in Synchronization.it(size):
            SL1 = loadmat(f"sync_{i}_{j}.mat")["data"].reshape(-1)
            SL2 = loadmat(f"sync_{j}_{i}.mat")["data"].reshape(-1)

            if (out[i] is None):
                out[i] = SL1
            else:
                out[i] += SL1

            if (out[j] is None):
                out[j] = SL2
            else:
                out[j] += SL2

        os.chdir("..")

        out = np.array(out) / (size-1)
        savemat("connectivity.mat", {"data": out})

        return out


if __name__ == "__main__":
    data = np.random.random((5, 2500))
    SL = Synchronization(data, 5, 2, 100, 410, 0.05)
    # SL.synchronization()
    connectivity = SL.connectivity()
    print(connectivity.shape)
    # print(output.size * output.itemsize)
