import sys
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
import unittest

import pulp.utils.barstest as barstest


class TestBarstest(unittest.TestCase):
    def setUp(self):
        self.H = 12
        self.R = self.H / 2
        self.D = self.R ** 2

    def test_generate(self):
        W = barstest.generate_bars(self.H)

        self.assertTrue(((W == 0.) + (W == 1.)).all())
        self.assertEqual(W.shape, (self.D, self.H))

    def test_generate_negbars(self):
        W = barstest.generate_bars(self.H, neg_bars=True)

        self.assertTrue(((W == -1.) + (W == 0.) + (W == 1.)).all())
        self.assertTrue((W == -1.).any())
        self.assertEqual(W.shape, (self.D, self.H))

    def test_find_permutation(self):
        H = self.H
        D = self.D

        Wgt = barstest.generate_bars(H)

        # perform random permutation
        for r in xrange(10):
            perm_gt = np.random.permutation(H)
            W = Wgt[:, perm_gt]

            self.assertEqual(W.shape, (D, H))

            perm = barstest.find_permutation(W, Wgt)
            perm2 = barstest.find_permutation2(W, Wgt)

            self.assertTrue((perm == perm2).all())

            self.assertTrue((W[:, perm] == Wgt).all())

    def test_overcomplete(self):
        H = self.H
        D = self.D

        Wgt = barstest.generate_bars(H)

        W = np.zeros((D, H + 2))
        W[:, :H] = Wgt[:, ::-1]

        perm = barstest.find_permutation(W, Wgt)

        self.assertTrue((W[:, perm] == Wgt).all())
