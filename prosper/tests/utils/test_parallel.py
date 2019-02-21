import numpy as np
import unittest

from mpi4py import MPI

import prosper.utils.parallel as parallel


class TestParallel(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD

    def test_allsort(self):
        D = 2
        my_N = 10
        N = self.comm.size * my_N

        my_a = np.random.uniform(size=(my_N, D))

        # Check axis=0,
        all_a = parallel.allsort(my_a,
                                 axis=0,
                                 kind='quicksort',
                                 comm=self.comm)
        self.assertEqual(all_a.shape, (N, D))

        # Chck default axis
        all_a = parallel.allsort(my_a, comm=self.comm)
        self.assertEqual(all_a.shape, (my_N, D * self.comm.size))

    def test_allargsort(self):
        D = 2
        my_N = 10
        N = self.comm.size * my_N

        my_a = np.random.uniform(size=(my_N, D))

        # Check axis=0,
        all_a = parallel.allargsort(my_a,
                                    axis=0,
                                    kind='quicksort',
                                    comm=self.comm)
        self.assertEqual(all_a.shape, (N, D))

        # Chck default axis
        all_a = parallel.allargsort(my_a, comm=self.comm)
        self.assertEqual(all_a.shape, (my_N, D * self.comm.size))

    def test_allmean(self):
        D = 10
        my_a = np.ones(D)

        mean = parallel.allmean(my_a)
        self.assertAlmostEqual(mean, 1.0)

    def test_allsum(self):
        D = 10
        my_a = np.ones(D)

        sum = parallel.allsum(my_a)
        self.assertAlmostEqual(sum, D * self.comm.size)
