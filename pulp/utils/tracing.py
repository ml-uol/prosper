#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""

Record tracepoint for runtime profiling/tracing.


Usage::
    import pulp.utils.tracing as tracing

    tracing.set_tracefile()

    tracing.tracepoint("Work:begin")
    # Do hard work
    tracing.tracepoint("Work:end")



ToDo: 
    Should change API so that "set_tracefile" takes a directory and 
    not a filename pattern.
"""

import os
import sys
import time
import os.path as path
import subprocess
from mpi4py import MPI
from functools import wraps

trace_fname = None
trace_file = None
start_time = None


def tracepoint(str):
    """ Record a tracepoint *str*.

    If tracing was enables with *set_tracefile* the given tracepoint will be
    recorded.
    """
    global trace_file
    global start_time

    if trace_file is None:
        return

    ts = MPI.Wtime() - start_time
    trace_file.write("[%f] [%s]\n" % (ts, str))
    #f = sys._getframe()
    #c = f.f_code
    #trace_file.write("[%f] [%s] %s:%d %s\n" % (ts, str, c.co_filename, f.f_lineno, c.co_name))


def traced(func):
    """ Decorator for functions to be traced.

    Whenever a traced function is called two tracepoints will be recorded:
    "func_name:begin" when the function starts to execute and "func_name:end"
    when the function is about to return.

    Usage:
    
        @traced
        def some_function_or_method(...):
            # do something
    """
    begin_str = func.func_name + ':begin'
    end_str = func.func_name + ':end'
    @wraps(func)
    def wrapped(*args, **kwargs):
        tracepoint(begin_str)
        res = func(*args, **kwargs)
        tracepoint(end_str)
        return res
    return wrapped


def set_tracefile(fname="trace-%04d.txt", comm=MPI.COMM_WORLD):
    """ Enable tracing

    The fname argument is expected to have a %d format specifier which will
    be replaced with the MPI rank.

    Has to be called on all rank simultaniously.
    """
    global trace_fname
    global trace_file
    global start_time

    trace_fname = fname
    fname = fname % comm.rank
    trace_file = open(fname, "w")
    trace_file.write("# Start time: %s\n" % time.asctime())
    trace_file.write("# Hostname: %s\n" % os.uname()[1])
    trace_file.write("# MPI size: %d rank: %d\n" % (comm.size, comm.rank))

    comm.Barrier(); start_time = MPI.Wtime()

def close(archive=True, comm=MPI.COMM_WORLD):
    """ Closes the tracefiles and archives all tracefiles
    """
    global trace_fname
    global trace_file
    global start_time

    # Tracing active at all?
    if trace_file is None:
        return

    # All ranks close the tracefile
    tracepoint("closing tracefiles")
    trace_file.close()
    comm.Barrier()

    if archive and comm.rank == 0:
        # Call external archiver..
        trace_dir, trace_tailname = path.split(trace_fname)
        trace_dir = path.normpath(trace_dir)

        archive_fname = path.join(trace_dir, "traces.tgz")
        archive_cmd = ["tar", "-czf", archive_fname, "-C", trace_dir]
        for rank in xrange(comm.size):
            archive_cmd.append(trace_tailname % rank)
        #print"Running ", archive_cmd
        subprocess.call(archive_cmd, stderr=subprocess.STDOUT)
        
        # Delete tracefiles
        rm_cmd = ["rm"]
        for rank in xrange(comm.size):
            rm_cmd.append(trace_fname % rank)
        #print"Running ", rm_cmd
        subprocess.call(rm_cmd, stderr=subprocess.STDOUT)
    
    # Cleanup global variables
    trace_fname = None
    trace_file = None
    start_time = None

