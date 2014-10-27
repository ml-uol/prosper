#
#  Lincense: Academic Free License (AFL) v3.0
#

"""
    Provide simple bar-pattern generator.
"""

from __future__ import division


__author__ = "Joerg Bornschein <bornschein@fias.uni-frankfurt.de>"

import numpy as np


def generate(num, size, p_bar):
    """
    Generate a set of *num* bar-patterns with *size* * *size* pixels. Each 
    of the 2* *size* possible bars is active with a probability of *p_bar*.
    The pixels of an active bar are set to 1, the pixels of the
    inactive bar are set to 0.

    This function returns the set of bar patterns as (N,size,size) shaped numpy
    array.

    You can easily add noise to the generated patterns
    by using the random number generators in :mod:`numpy.random`.

    Example::

        import numpy as np
        from pulp.utils import bargen

        N = 1000; size = 16
        bars = bargen.generate_pattern(N, size, 2./15)
        bars = 10*bars + 5*np.random.randn( (N,size,size) )
        bars = bars.reshape(N, size*size)

    Generates 1000 bar-patterns in the interval [0,10] with additive gaussian
    noise of  #math`\sigma=5` variance.
    """

    bars = np.zeros( (num, size, size) )

    for n in xrange(num):
        for i in range(size):
            # generate horizontal bar
            if np.random.random() <= p_bar:
                bars[n, i, :] = 1.
            # generate vertical bar
            if np.random.random() <= p_bar:
                bars[n, :, i] = 1.

    return bars

#For unit-test purposes
def W_gen(H, bar_val):
    """Returns (*H*, (H//2)**2) array with bars inside of activity *bar_val*"""
    D2 = H//2
    D = D2**2
    W_gt = np.zeros( (H, D2, D2) )
    for i in xrange(D2):
        W_gt[   i, i, :] = bar_val
        W_gt[D2+i, :, i] = bar_val
    return W_gt.reshape( (H, D) )
    
