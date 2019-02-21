from __future__ import division

import numpy as np

#=============================================================================
# Generate basis

def generate_bars_dict(H, neg_bars=False):
    """ Generate a ground-truth dictionary W suitable for a std. bars test

    Creates H bases vectors with horizontal and vertival bars on a R*R pixel grid,
    (wth R = H // 2).  The function thus returns a matrix storing H dictionaries of 
    size D=R*R.

    :param H: Number of latent variables
    :type  H: int
    :param neg_bars: Should half the bars have a negative value (-1)?
    :type  neg_bars: bool
    :rtype: ndarray (D x H)

    """
    R = H // 2
    D = R ** 2
    W_gt = np.zeros((R, R, H))
    for i in xrange(R):
        W_gt[i, :, i] = 1.
        W_gt[:, i, R + i] = 1.

    if neg_bars:
        sign = 1 - 2 * np.random.randint(2, size=(H, ))
        W_gt = sign[None, None, :] * W_gt
    return W_gt.reshape((D, H))

def generate_bars_data(num, size, p_bar):
    """ Generate a bars-test dataset.

    Create a dataset consisting of *num* datapoints, each a size*size pixel grid.
    The individual bars have a i.i.d. probability of being active of *p_bar*.
    """
    data = np.zeros((num, size, size))

    for n in xrange(num):
        for i in range(size):
            # generate horizontal bar
            if np.random.random() <= p_bar:
                data[n, i, :] = 1.
            # generate vertical bar
            if np.random.random() <= p_bar:
                data[n, :, i] = 1.

    return data.reshape(num, size*size)

#=============================================================================
# Find permutations

def find_permutation(W, Wgt):
    """ Check if *W* is a permutated version of *Wgt* and return the permutation.
  
    :param W:   A matrix containing *H'* dictionay elements.
    :type  W:   ndarray (D x H')
    :param Wgt: A matrix containing *H* dictionary elements
    :type  Wgt: ndarray (D x H)
    :rtype: ndarray (dtype=int)

    This implementation uses a greedy assignment algorithm with runtime $O(H H' D)$.
    """

    D, H = W.shape
    Dgt, Hgt = Wgt.shape

    assert D == Dgt
    assert H >= Hgt

    MAEs = np.zeros((Hgt, H))

    for i in xrange(Hgt):
        for j in xrange(H):
            MAEs[i, j] = np.sum(np.abs(Wgt[:, i] - W[:, j])) / D

    perm = np.zeros(Hgt, dtype=np.int)

    total_mae = 0.
    for k in xrange(Hgt):
        # Find best matching combination
        pos = np.argmin(MAEs)

        i = pos // H  # 0 <= i <= Hgt
        j = pos % H  # o <= j <= H

        total_mae += MAEs[i, j]
        perm[i] = j

        MAEs[i, :] = np.inf
        MAEs[:, j] = np.inf

    total_mae = total_mae / Hgt

    return perm


def find_permutation2(W, Wgt):
    """Check if *W* is a permutated version of *Wgt*.

    :param W:   A matrix containing *H'* dictionay elements.
    :type  W:   ndarray (D x H')
    :param Wgt: A matrix containing *H* dictionary elements
    :type  Wgt: ndarray (D x H)
    :rtype: ndarray (dtype=int)

    This implementation uses dynamic programming to assignment algorithm with runtime > O(H H' D).
    """
    D, H = W.shape
    Dgt, Hgt = Wgt.shape

    assert D == Dgt

    # Calculate error matrix
    error = np.zeros((Hgt, H))

    for i in xrange(Hgt):
        for j in xrange(H):
            error[i, j] = np.sum(np.abs(Wgt[:, i] - W[:, j])) / D

    # Allocate tables for dynamic programming
    mae_tab = np.zeros((Hgt, H))
    used_tab = np.empty((Hgt, H), dtype=np.object)

    # Initialize first row
    for ht in xrange(H):
        tmprow = error[0, :].copy()
        tmprow[ht] = np.inf

        minpos = np.argmin(tmprow)
        mae_tab[0, ht] = error[0, minpos]
        used_tab[0, ht] = [minpos]

        # Build table
    for hr in xrange(1, Hgt - 1):
        for ht in xrange(H):

            # Build Matrix
            tmpmtx = np.zeros((Hgt, H))
            for h0 in xrange(Hgt):
                for h1 in xrange(H):
                    val = mae_tab[hr - 1, h1] + error[hr, h0]
                    if h0 in used_tab[hr - 1, h1]:
                        val = np.inf
                    if ht in used_tab[hr - 1, h1]:
                        val = np.inf
                    if ht == h0:
                        val = np.inf
                    tmpmtx[h0, h1] = val

            minpos_error = np.argmin(tmpmtx) // Hgt
            minpos_prev = np.argmin(tmpmtx) % Hgt

            mae_tab[hr, ht] = tmpmtx[minpos_error, minpos_prev]
            used_tab[hr, ht] = used_tab[hr - 1, minpos_prev] + [minpos_error]

    # Last row
    for ht in xrange(H):
        mae_tab[-1, ht] = mae_tab[-2, ht] + error[-1, ht]
        used_tab[-1, ht] = used_tab[-2, ht] + [ht]

    # Normalize MAE
    mae_tab /= Hgt

    minpos = np.argmin(mae_tab[-1, :])
    return np.array(used_tab[-1, minpos])
