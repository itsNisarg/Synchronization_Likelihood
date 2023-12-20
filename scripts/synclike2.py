""" This module contains the low memory implementation of the Synchronization Likelihood Algorithm."""

import sys
import os
import numpy as np
import scipy as sp
from scipy.io import savemat, loadmat
from numba import njit


class Synchronization(object):
    """Synchronization Likelihood Algorithm.
    Parameters
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
        Dictionary containing the critical epsilon for each time series.
    data : ndarray
        Time series.

    Methods
    -------
    _get_embeddings(x, m, lag)
        Embedding of a time series x with embedding dimension m and lag l.
    _get_distance(Xi, Xj)
        Compute the distance between two points.
    _get_distances(X)
        Compute the distances between all pairs of points in X.
    _get_prob(X, eps, i, w1, w2)
        Compute the probability of a point i being in the epsilon-neighborhood
        of another point j.
    _obj(eps, X, i, w1, w2, pref)
        Objective function for the root finding algorithm.
    _get_critical_eps(X, pref, w1, w2, brack)
        Compute the critical epsilon for a given preferred probability.
    _get_Hij(X1, X2, E1, E2, w1, w2)
        Compute the H matrix.
    _calc_SL(X, E, H)
        Compute the synchronization likelihood matrix.
    _avg(sl, w1, w2)
        Compute the average synchronization likelihood.
    SyncLike(x, y)
        Compute the synchronization likelihood for two time series.
    synchronization()
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
        self.data = data

    @staticmethod
    @njit
    def _get_embeddings(x, m, lag):
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
        X = np.zeros((m, size))
        for i in range(m):
            X[i] = x[i * lag: i * lag + size]

        return X.T

    @staticmethod
    @njit
    def _get_distance(Xi, Xj):
        """Compute the distance between two points.
        Parameters
        ----------
        Xi : ndarray
            Point.
        Xj : ndarray
            Point.
        """
        return np.linalg.norm(Xi - Xj)

    @staticmethod
    @njit
    def _get_distances(X):
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
    def _get_prob(X, eps, i, w1, w2):
        """Compute the probability of a point i being in the epsilon-neighborhood
        of another point j.
        Parameters
        ----------
        X : ndarray
            Embedding of a time series.
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
            Probability of a point i being in the epsilon-neighborhood of another point j.
        """
        summ = 0
        for j in range(len(X)):
            if ((w1 < abs(i - j) < w2) and (Synchronization._get_distance(X[i], X[j]) < eps)):
                summ += 1
        return summ / (2 * (w2 - w1))

    @staticmethod
    def _obj(eps, X, i, w1, w2, pref):
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
        return pref - Synchronization._get_prob(X, eps, i, w1, w2)

    @staticmethod
    def _get_critical_eps(X, pref, w1, w2, brack):
        """Compute the critical epsilon for a given preferred probability.
        Parameters
        ----------
        X : ndarray
            Embedding of a time series.
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
        e = np.zeros(len(X), dtype=np.float32)

        for i in range(len(X)):
            try:
                result = sp.optimize.root_scalar(Synchronization._obj, args=(
                    X, i, w1, w2, pref), bracket=brack, method='brentq')
                if (not result.converged):
                    raise Exception("Root finding algorithm did not converge")
                e[i] = result.root
            except Exception as e:
                print(e)
                sys.exit(1)
        return e

    @staticmethod
    def _get_Hij(X1, X2, E1, E2, w1, w2):
        """Compute the H matrix.
        Parameters
        ----------
        X1 : ndarray
            Embedding of a time series.
        X2 : ndarray
            Embedding of a time series.
        E1 : ndarray
            Critical epsilon for X1.
        E2 : ndarray
            Critical epsilon for X2.

        Returns
        -------
        hij : ndarray
            H matrix.
        """
        hij = np.zeros((len(X1), len(X2)), dtype=np.int16)

        for i in range(len(X1)):
            for j in range(len(X2)):
                if (w1 < abs(i - j) < w2):
                    hij[i, j] = (Synchronization._get_distance(X1[i], X1[j]) < E1[i]) + \
                        (Synchronization._get_distance(X2[i], X2[j]) < E2[i])

        return hij

    @staticmethod
    def _calc_SL(X, E, H):
        """Compute the synchronization likelihood matrix.
        Parameters
        ----------
        X : ndarray
            Embedding of a time series.
        E : ndarray
            Critical epsilon.
        H : ndarray
            H matrix.

        Returns
        -------
        Sij : ndarray
            Synchronization likelihood matrix.
        """
        Sij = np.zeros((len(X), len(X)), dtype=np.float32)

        for i in range(len(X)):
            for j in range(len(X)):
                Sij[i, j] = H[i, j] - \
                    1 if (Synchronization._get_distance(
                        X[i], X[j]) < E[i]) else 0

        return Sij

    @staticmethod
    @njit
    def _avg(sl, w1, w2):
        """Compute the average synchronization likelihood.
        Parameters
        ----------
        SL : ndarray
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

    def SyncLike(self, x, y):
        """Compute the synchronization likelihood for two time series.
        Parameters
        ----------
        x : ndarray
            Time series.
        y : ndarray
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

        Returns
        -------
        SL1 : ndarray
            Average synchronization likelihood for x.
        SL2 : ndarray
            Average synchronization likelihood for y.
        """

        m, lag, w1, w2, pref = self.m, self.lag, self.w1, self.w2, self.pref

        X1 = Synchronization._get_embeddings(self.data[x], m, lag)
        X2 = Synchronization._get_embeddings(self.data[y], m, lag)

        # print("SyncLike: Computing Distances...")
        # D1 = _get_distances(X1)
        # D2 = _get_distances(X2)
        # print("SyncLike: Computing Distances Done")

        print("SyncLike: Computing Critical Epsilon...")
        if (x not in self.E):
            E1 = Synchronization._get_critical_eps(
                X1, pref, w1, w2, [0.0, 1])
            self.E[x] = E1
        else:
            E1 = self.E[x]

        if (y not in self.E):
            E2 = Synchronization._get_critical_eps(
                X2, pref, w1, w2, [0.0, 1])
            self.E[y] = E2
        else:
            E2 = self.E[y]
        print("SyncLike: Computing Critical Epsilon Done")

        print("SyncLike: Computing H...")
        H = Synchronization._get_Hij(X1, X2, E1, E2, w1, w2)
        print("SyncLike: Computing H Done")

        print("SyncLike: Computing SL...")
        SLij1 = Synchronization._calc_SL(X1, E1, H)
        SLij2 = Synchronization._calc_SL(X2, E2, H)
        print("SyncLike: Computing SL Done")

        print("SyncLike: Computing Average SL...")
        SL1 = Synchronization._avg(SLij1, w1, w2)
        SL2 = Synchronization._avg(SLij2, w1, w2)
        print("SyncLike: Computing Average SL Done")

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

        if (not os.path.exists("./sync2")):
            os.mkdir("sync2")
        os.chdir("sync2")
        for i, j in Synchronization.it(size):
            # out[i][j], out[j][i] = Synchronization.SyncLike(self, i, j)
            print(f"SAVING {i} {j}")
            SL1, SL2 = Synchronization.SyncLike(self, i, j)
            savemat(f"sync_{i}_{j}", {"data": SL1})
            savemat(f"sync_{j}_{i}", {"data": SL2})

        os.chdir("..")
        print(os.getcwd())

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
    SL.synchronization()
    # print(output.size * output.itemsize)
