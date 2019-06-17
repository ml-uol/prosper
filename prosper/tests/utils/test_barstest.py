


# import sys
# sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
import unittest

import prosper.utils.barstest as barstest


class TestBarstest(unittest.TestCase):
    def setUp(self):
        self.H = 12
        self.R = self.H // 2
        self.D = self.R ** 2

    def test_generate(self):
        W = barstest.generate_bars_dict(self.H)

        self.assertTrue(((W == 0.) + (W == 1.)).all())
        self.assertEqual(W.shape, (self.D, self.H))

    def test_generate_negbars(self):
        W = barstest.generate_bars_dict(self.H, neg_bars=True)

        self.assertTrue(((W == -1.) + (W == 0.) + (W == 1.)).all())
        self.assertTrue((W == -1.).any())
        self.assertEqual(W.shape, (self.D, self.H))

    def test_find_permutation(self):
        H = self.H
        D = self.D

        Wgt = barstest.generate_bars_dict(H)

        # perform random permutation
        for r in range(10):
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

        Wgt = barstest.generate_bars_dict(H)

        W = np.zeros((D, H + 2))
        W[:, :H] = Wgt[:, ::-1]

        perm = barstest.find_permutation(W, Wgt)

        self.assertTrue((W[:, perm] == Wgt).all())

    def test_generate_bars_data(self):
        num = 100
        size = 5
        p_bar = 1. / size

        data = barstest.generate_bars_data(100, size, p_bar)

        assert data.shape == (num, size*size)
