#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#

"""
  Demonstrate how to use the annealing class to generate 
  piecewise-linear annealing schedules.
"""

from __future__ import division

import sys
sys.path.insert(0, '..')

from pulp.em.annealing import LinearAnnealing

Tsteps = 80
Tstart = 20
Tend = 1.05


# Choose annealing schedule
anneal = LinearAnnealing(Tsteps)

anneal['T'] = [(10, Tstart) , (-10, Tend)]
anneal['param_a'] = [(2/3, 0.) , (-10, 1.)]
anneal['param_b'] = 0.5
    
assert anneal['param_c'] == 0.0

while not anneal.finished:
    print "[%3d] T=%.2f   parameter_a=%.2f     parameter_b=%.2f" % (anneal['step'], anneal['T'], anneal['param_a'], anneal["param_b"])
    anneal.next()

