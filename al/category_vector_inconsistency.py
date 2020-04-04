from numba import njit
import numpy as np

# Paper reference: https://www.sciencedirect.com/science/article/abs/pii/S0925231217313371?via%3Dihub


@njit
def single_entropy(w):
    """
    Calculate entropy for single vector
    Args:
        w: A probability value between 0 and 1

    Returns:

    """
    if w == 0:
        return -(1-w)*np.log2(1-w)
    elif w == 1:
        return -w*np.log2(w)
    else:
        return -(w*np.log2(w)+(1-w)*np.log2(1-w))


@njit
def pair_entropy(U, T):
    """
    Calculate entropy between two 1D vectors for equal size
    Args:
        U: binarized unlabelled predictions
        T: Ground truth labels from the training set

    Returns:

    """
    q = U.shape[0]
    a = np.logical_and(U, T).sum()
    b = np.logical_and(U, np.logical_not(T)).sum()
    c = np.logical_and(np.logical_not(U), T).sum()
    d = np.logical_and(np.logical_not(U), np.logical_not(T)).sum()
    return single_entropy((b + c) / q) + ((b + c) / q) * single_entropy(b / (b + c)) + ((a + d) / q) * single_entropy(
        a / (a + d))


@njit
def de(U, T):
    """
    Calculates normalized entropy distance between two 1D vectors for equal size
    Args:
        U: binarized unlabelled predictions
        T: Ground truth labels from the training set

    Returns:

    """
    q = U.shape[0]
    return 2 - (single_entropy(U.sum() / q) + single_entropy(T.sum() / q)) / pair_entropy(U, T)


@njit
def dh(U, T):
    """
    Hamming distance between two 1D binary vectors of the same size
    Args:
        U: binarized unlabelled predictions
        T: Ground truth labels from the training set

    Returns:

    """
    q = U.shape[0]
    b = np.logical_and(U, np.logical_not(T)).sum()
    c = np.logical_and(np.logical_not(U), T).sum()
    return (b + c) / q


@njit
def single_cvi(U, T):
    """
    Category vector inconsistency between two 1D binary vectors of the same size
    Args:
        U: binarized unlabelled predictions
        T: Ground truth labels from the training set

    Returns:

    """
    if dh(U, T) == 1:
        return 1
    elif dh(U, T) == 0:
        return 0
    else:
        return de(U, T)


@njit(parallel=True)
def cvi(U_T, T_T):
    """
    Cumulative Category vector inconsistency between two 2D matrix of different sizes
    Args:
        U_T: binarized unlabelled predictions
        T_T: Ground truth labels from the training set

    Returns:

    """
    Ls = T_T.shape[0]
    Us = U_T.shape[0]
    fu = np.zeros(Us)
    for i in prange(Us):
        dist = np.zeros(Ls)
        for j in range(Ls):
            dist[j] = single_cvi(U_T[i], T_T[j])

        fu[i] = dist.sum() / Ls
    return fu
