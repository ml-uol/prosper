#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""
    The pulp.utils packages provides various utility functions and classes.
"""
import time as tm
import numpy as np
import errno
import sys
import os

from mpi4py import MPI

def create_output_path(basename=None):
    """ Creates the output directory making sure you don't overwrite a folder. 

    If you are running under the torque/pbs job scheduler, the directory will
    be named according to <BASENAME>.d<JOBID>; if you are not, the directory
    will be named <BASENAME>.<DATE+TIME>. If such a directory should already
    exist, an additional "+NUMBER" suffix will be appended.
    
    If you do not specify a *basename*, it will derive the base-name of the 
    directory from your programs name (sys.argv[0]).

    Returns the path of the newly created directory.
    """
    comm = MPI.COMM_WORLD

    if comm.rank == 0:                     # MPI Rank 0 does all the work
        if basename is None:
            basename = sys.argv[0]

        # Determine suffix
        if 'PBS_JOBID' in os.environ:
            job_no = os.environ['PBS_JOBID'].split('.')[0]   # Job Number
            suffix = "d"+job_no
        elif 'SLURM_JOBID' in os.environ:
            job_no = os.environ['SLURM_JOBID'] 
            suffix = "d"+job_no
        else:
            suffix = tm.strftime("%Y-%m-%d+%H:%M")
            
        suffix_counter = 0
        dirname = "output/%s.%s" % (basename, suffix)
        while True:
            try:
                os.makedirs(dirname)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise e
                suffix_counter += 1
                dirname = "output/%s.%s+%d" % (basename, suffix, suffix_counter)
            else:
                break
    else:
        dirname = None
        
    return comm.bcast(dirname)+"/"

