
== Introduction ==

This package contains all the source code to reproduce the numerical
experiments described in the paper. It contains a parallelized implementation
of the Binary Sparse Coding (BSC) and Maximum Causes Analysis (MCA) generative
models training algorithm.


== Dependencies ==

 * NumPy
 * SciPy
 * mpi4py (>= 1.3)

== Overview ==

pulp/       - Python library/framework for MPI parallelized 
               EM-based algorithms. The MCA implementation
               can be found in pulp/em/camodels/mmca_et.py, the 
               BSC implementation in pulp/em/camodels/bsc_et.py.

examples/   - Small example programs for the ulp library



== Software dependencies ==
 
 * Python (>= 2.6)
 * NumPy (reasonably recent)
 * SciPy (reasonably recent)
 * pytables (reasonably recent)
 * mpi4py (>= 1.2)
 
== Running ==

To run some toy examples:

  $ cd examples/barstest
  $ python bars-run-all.py param-bars-<...>.py

where <...> should be appropriately replaced to correspond to one of the parameter 
files available in the directory. The bars-run-all.py script should then initialize 
and run the algorithm which corresponds to the chosen parameter file. 

== Results/Output ==

The results produced by the code are stored in a 'results.h5' file 
under "./output/.../". The file stores the model parameters (e.g., W, pi etc.) 
for each EM iteration performed. To read the results file, you can use
openFile function of the standard tables package in python. Moreover, the
results files can also be easily read by other packages such as Matlab etc.

== Running on a parallel architecture ==

The code uses MPI based parallelization. If you have parallel resources
(i.e., a multi-core system or a compute cluster), the provided code can make a 
use of parallel compute resources by evenly distributing the training data 
among multiple cores.

To run the same script as above, e.g., 

a) On a multi-core machine with 32 cores:

 $ mpirun -np 32 python bars-run-all.py param-bars-<...>.py

b) On a cluster:

 $ mpirun --hostfile machines python bars-run-all.py param-bars-<...>.py

 where 'machines' contains a list of suitable machines.

See your MPI documentation for the details on how to start MPI parallelized 
programs.



