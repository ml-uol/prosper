***************************************
Expectation Maximization infrastructure
***************************************
The machine learning algorithms distributed with ProSper are based on latent variable probabilistic data models and are trained with variational Expectation Maximization (EM) learning algorithm. The source code is therefore organized under an EM module.

Expectation Maximization Algorithm
==================================
The Expectation Maximization (EM) algorithm is used to optimize probabilistic models with latent random variables.
It is an iterative algorithm that optimizes with respect to the posterior distribution in the E-step and and proceeds with an optimization of the model parameters in the M step. A simple implementation is given in the EM class. The more technical components however are contained in the Model classes. 

.. automodule:: prosper.em
	:members:

Component Analysis Models
=========================
Component Analysis Models refers to models with multiple latent variables may contribute to the same datapoints to 

.. automodule:: prosper.em.camodels
	:members:

Binary Sparse Coding
--------------------

.. autoclass:: prosper.em.camodels.bsc_et.BSC_ET
    :members:

Ternary Sparse Coding
---------------------

.. autoclass:: prosper.em.camodels.tsc_et.TSC_ET
	:members:

Discrete Sparse Coding
----------------------

.. autoclass:: prosper.em.camodels.dsc_et.DSC_ET
	:members:

Spike-and-Slab Sparse Coding
----------------------------

.. autoclass:: prosper.em.camodels.gsc_et.GSC
	:members:

Maximum Component Analysis
--------------------------

.. autoclass:: prosper.em.camodels.mca_et.MCA_ET
	:members:

Maximum Magnitude Component Analysis
------------------------------------
.. autoclass:: prosper.em.camodels.mmca_et.MMCA_ET
	:members:


Mixture Models
==============
Mixture Models refers to models where a single latent variable is responsible for the generation of a datapoint 

.. automodule:: prosper.em.mixturemodels
	:members:


Mixture of Gaussians
--------------------
Standard mixture of Gaussians model:

.. automodule:: prosper.em.mixturemodels.MoG
    :members:

Mixture of Gaussians
--------------------
Standard mixture model with a Poisson observation noise model:

.. automodule:: prosper.em.mixturemodels.MoP
	:members:


Annealing
=========
The annealing module holds utilities relevant to making minor
modifications to the training process.


Annealing Class
---------------
This is a generic class inheritted by all annealing objects:

.. autoclass:: prosper.em.annealing.Annealing
	:members:

Linear Annealing
----------------
The linear annealing class is an example annealing class that makes changes at hyperparameters changing with linear rate over EM iterations.

.. autoclass:: prosper.em.annealing.LinearAnnealing
	:members:

