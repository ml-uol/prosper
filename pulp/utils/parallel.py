#
#  Lincense: Academic Free License (AFL) v3.0
#

from __future__ import division

import sys 
import os
import numpy as np
from mpi4py import MPI 

#=============================================================================
# Parallel & pretty printer

typemap = {
    np.dtype('float64'): MPI.DOUBLE,
    np.dtype('float32'): MPI.FLOAT,
    np.dtype('int16'): MPI.SHORT,
    np.dtype('int32'): MPI.INT,
    np.dtype('int64'): MPI.LONG,
    np.dtype('uint16'): MPI.UNSIGNED_SHORT,
    np.dtype('uint32'): MPI.UNSIGNED_INT,
    np.dtype('uint64'): MPI.UNSIGNED_LONG,
}


def pprint(obj="", comm=MPI.COMM_WORLD, end='\n'):
    """ 
    Parallel print: Make sure only one of the MPI processes
    calling this function actually prints something. All others
    (comm.rank != 0) return without doing enything.
    """
    if comm.rank != 0:
        return 

    if isinstance(obj, str):
        sys.stdout.write(obj+end)
    else:
        sys.stdout.write(repr(obj))
        sys.stdout.write(end)
        sys.stdout.flush()

def stride_data(N, balanced=False, comm=MPI.COMM_WORLD):
    """ Stride data 

    Calculates a suitable (block-) distribution for *N* data items
    for *comm.size* parallel workers.
    
    :param N: total number of data items to be sharded
    :type  N: int
    :param balanced: do all ranks receive he same number of data-items? 
        Resisuums will be silently discarded! (default False)
    :type  balanced: bool
    :returns: (first, last) The first and last item this 
    :rtype: (int, int)

    Example::
        
        comm = MPI.COMM_WORLD
        big_array = np.arange(100000)            
    
        my_firsy, my_last = parallel.stride_data(big_array.size)     # partition data 
        my_array = big_array[my_first:my_last]

    """
    my_N = N // comm.size
    residue = N % comm.size

    if balanced:
        first = my_N * comm.rank
        last  = my_N * (comm.rank+1)
        return first, last
    else:
        if comm.rank < residue:
            size = my_N+1
            first = size*comm.rank
            last  = first+size 
        else:
            size = my_N
            first = size*comm.rank+residue
            last = first+size

    return first, last

def allsort(my_array, axis=-1, kind='quicksort', order=None, comm=MPI.COMM_WORLD):
    """
    Parallel (collective) version of numpy.sort
    """
    shape = my_array.shape
    all_shape = list(shape)
    all_shape[axis] = comm.allreduce(shape[axis])

    if not my_array.dtype in typemap:
        raise TypeError("Dont know how to handle arrays of type %s" % my_array.dtype)
    mpi_type = typemap[my_array.dtype]

    my_sorted = np.sort(my_array, axis, kind, order)

    all_array = np.empty(shape=all_shape, dtype=my_sorted.dtype)
    comm.Allgather((my_sorted, mpi_type), (all_array, mpi_type))

    all_sorted = np.sort(all_array, axis, 'mergesort', order)
    return all_sorted

def allargsort(my_array, axis=-1, kind='quicksort', order=None, comm=MPI.COMM_WORLD):
    """ Parallel (collective) version of numpy.argsort
    """
    shape = my_array.shape
    all_shape = list(shape)
    all_shape[axis] = comm.allreduce(shape[axis])

    if not my_array.dtype in typemap:
        raise TypeError("Dont know how to handle arrays of type %s" % my_array.dtype)
    mpi_type = typemap[my_array.dtype]

    my_sorted = np.argsort(my_array, axis, kind, order)

    all_array = np.empty(shape=all_shape, dtype=my_sorted.dtype)
    comm.Allgather((my_sorted, mpi_type), (all_array, mpi_type))

    all_sorted = np.argsort(all_array, axis, kind, order)
    return all_sorted

def allmean(my_a, axis=None, dtype=None, out=None, comm=MPI.COMM_WORLD):
    """ Parallel (collective) version of numpy.mean
    """
    shape = my_a.shape
    if axis is None:
        N = comm.allreduce(my_a.size)
    else:
        N = comm.allreduce(shape[axis])

    my_sum = np.sum(my_a, axis, dtype)

    if my_sum is np.ndarray:
        sum = np.empty_like(my_sum)
        comm.Allreduce( (my_sum, typemap[my_sum.dtype]), (sum, typemap[sum.dtype]))
        sum /= N
        return sum
    else:
        return comm.allreduce(my_sum) / N

def allsum(my_a, axis=None, dtype=None, out=None, comm=MPI.COMM_WORLD):
    """ Parallel (collective) version of numpy.sum
    """
    my_sum = np.sum(my_a, axis, dtype)

    if my_sum is np.ndarray:
        sum = np.empty_like(my_sum)
        comm.Allreduce( (my_sum, typemap[my_sum.dtype]), (sum, typemap[sum.dtype]))
        return sum
    else:
        return comm.allreduce(my_sum)

