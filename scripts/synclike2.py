""" This module contains the low memory implementation of the Synchronization Likelihood Algorithm."""

import sys
import numpy as np
import scipy as sp
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
    """

    def __init__(self, data, m, lag, w1, w2, pref) -> None:
        self.m, self.lag, self.w1, self.w2, self.pref = m, lag, w1, w2, pref
        self.E = dict()
        self.D = dict()
        self.X = dict()
        self.data = data

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

    @njit
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
            if ((w1 < abs(i - j) < w2) and (_get_distance(X[i], X[j]) < eps)):
                summ += 1
        return summ / (2 * (w2 - w1))

    @njit
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
        return pref - _get_prob(X, eps, i, w1, w2)

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
                result = sp.optimize.root_scalar(_obj, args=(
                    X, i, w1, w2, pref), bracket=brack, method='brentq', maxiter=50)
                if (not result.converged):
                    raise Exception("Root finding algorithm did not converge")
                e[i] = result.root
            except Exception as e:
                print(e)
                sys.exit(1)

        return e

    @njit
    def _get_Hij(X1, X2, E1, E2):
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
                hij[i, j] = (_get_distance(X1[i], X1[j]) < E1[i]) + \
                    (_get_distance(X2[i], X2[j]) < E2[i])

        return hij

    @njit
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
                    1 if (_get_distance(X[i], X[j]) < E[i]) else 0

        return Sij

    @njit
    def _avg(SL, w1, w2):
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
        S = np.zeros(len(SL), dtype=np.float32)

        for i in range(len(SL)):
            summ = 0
            for j in range(len(SL)):
                if (w1 < abs(i - j) < w2):
                    summ += SL[i, j]
            S[i] = summ / (2 * (w2 - w1))

        return S

    def SyncLike(x, y, m, lag, w1, w2, pref):
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
        X1 = _get_embeddings(x, m, lag)
        X2 = _get_embeddings(y, m, lag)

        # print("SyncLike: Computing Distances...")
        # D1 = _get_distances(X1)
        # D2 = _get_distances(X2)
        # print("SyncLike: Computing Distances Done")

        print("SyncLike: Computing Critical Epsilon...")
        E1 = _get_critical_eps(X1, pref, w1, w2, [0.0, 0.01])
        E2 = _get_critical_eps(X2, pref, w1, w2, [0.0, 0.01])
        print("SyncLike: Computing Critical Epsilon Done")

        print("SyncLike: Computing H...")
        H = _get_Hij(X1, X2, E1, E2)
        print("SyncLike: Computing H Done")

        print("SyncLike: Computing SL...")
        SLij1 = _calc_SL(X1, E1, H)
        SLij2 = _calc_SL(X2, E2, H)
        print("SyncLike: Computing SL Done")

        print("SyncLike: Computing Average SL...")
        SL1 = _avg(SLij1, w1, w2)
        SL2 = _avg(SLij2, w1, w2)
        print("SyncLike: Computing Average SL Done")

        return (SL1 + SL2) / (2 * pref)
