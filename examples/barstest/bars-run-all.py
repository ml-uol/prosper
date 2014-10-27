#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

from mpi4py import MPI

from pulp.utils import create_output_path 
from pulp.utils.parallel import pprint

if len(sys.argv) != 2:
    pprint ("Usage %s <param-file>" % sys.argv[0])
    exit(1)

param_file = sys.argv[1]
output_path = create_output_path()

def rank0only(func, *args, **kargs):
    if MPI.COMM_WORLD.rank == 0:
        func(*args, **kargs)
    MPI.COMM_WORLD.barrier()

# make sure only one worker create data
rank0only(execfile, "./bars-create-data.py")

# learning is parallel on all MPI ranks
execfile("./bars-learning.py")



