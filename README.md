
[![Build Status](https://api.shippable.com/projects/557c833cedd7f2c05214da81/badge?branchName=master)](https://app.shippable.com/projects/557c833cedd7f2c05214da81/builds/master)
[![Documentation](https://readthedocs.org/projects/parallel-em/badge/?version=latest)](http://parallel-em.readthedocs.org/en/latest/)

Introduction
============

This package contains all the source code to reproduce the numerical
experiments described in the paper. It contains a parallelized implementation
of the Binary Sparse Coding (BSC) [1], Gaussian Sparse Coding (GSC) [2], 
Maximum Causes Analysis (MCA) [3], Maximum Magnitude Causes Analysis (MMCA) [4] and 
Ternary Sparse Coding (TSC) [5] models. All these probabilistic generative models 
are trained using a truncated Expectation Maximization (EM) algorithm [6].


Software dependencies
=====================
 
 * Python (>= 2.6)
 * NumPy (reasonably recent)
 * SciPy (reasonably recent)
 * pytables (reasonably recent)
 * mpi4py (>= 1.3)

Overview
========

prosper/       - Python library/framework for MPI parallelized 
              EM-based algorithms. The models' implementations
              can be found in prosper/em/camodels/.

examples/   - Small examples for initializing and running the models


Running
=======

To run some toy examples:

```
  $ cd examples/barstest
  $ python bars-run-all.py param-bars-<...>.py
```

where <...> should be appropriately replaced to correspond to one of the parameter 
files available in the directory. The bars-run-all.py script should then initialize 
and run the algorithm which corresponds to the chosen parameter file. 


Results/Output
==============

The results produced by the code are stored in a 'results.h5' file 
under "./output/.../". The file stores the model parameters (e.g., W, pi etc.) 
for each EM iteration performed. To read the results file, you can use
openFile function of the standard tables package in python. Moreover, the
results files can also be easily read by other packages such as Matlab etc.


Running on a parallel architecture
==================================

The code uses MPI based parallelization. If you have parallel resources
(i.e., a multi-core system or a compute cluster), the provided code can make a 
use of parallel compute resources by evenly distributing the training data 
among multiple cores.

To run the same script as above, e.g., 

a) On a multi-core machine with 32 cores:

 `$ mpirun -np 32 python bars-run-all.py param-bars-<...>.py`

b) On a cluster:

 `$ mpirun --hostfile machines python bars-run-all.py param-bars-<...>.py`

 where 'machines' contains a list of suitable machines.

See your MPI documentation for the details on how to start MPI parallelized 
programs.


References
==========

[1] M. Henniges, G. Puertas, J. Bornschein, J. Eggert, and J. Lücke (2010).
Binary Sparse Coding.
Proc. LVA/ICA 2010, LNCS 6365, 450-457. 

[2] A.-S. Sheikh, J. A. Shelton, J. Lücke (2014).
A Truncated EM Approach for Spike-and-Slab Sparse Coding.
Journal of Machine Learning Research, 15:2653-2687. 

[3] G. Puertas, J. Bornschein, and J. Lücke (2010). 
The Maximal Causes of Natural Scenes are Edge Filters.
Advances in Neural Information Processing Systems 23, 1939-1947. 

[4] J. Bornschein, M. Henniges, J. Lücke (2013).
Are V1 simple cells optimized for visual occlusions? A comparative study.
PLOS Computational Biology 9(6): e1003062.

[5] G. Exarchakis, M. Henniges, J. Eggert, and J. Lücke (2012).
Ternary Sparse Coding.
International Conference on Latent Variable Analysis and Signal Separation (LVA/ICA), 204-212. 

[6] J. Lücke and J. Eggert (2010). 
Expectation Truncation and the Benefits of Preselection in Training Generative Models.
Journal of Machine Learning Research 11:2855-2900. 

[5] G. Exarchakis, and J. Lücke (2017).
Discrete Sparse Coding.
Neural Computation, 29(11), 2979-3013.