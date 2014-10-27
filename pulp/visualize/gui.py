#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

import numpy as np
from math import sqrt

from mpi4py import MPI
from pulp.utils.datalog import dlog, GUIDataHandler
from tables import openFile
from os import mkdir, environ

# Make sure only MPI rank 0 imports GUI libraries...
if MPI.COMM_WORLD.rank == 0:
    import matplotlib as mpl
    mpl.use('GTKAgg')

    from matplotlib import pyplot
    import pylab as pl

    # Check if GUI is available at all...
    if not 'DISPLAY' in environ:
        raise EnvironmentError('FATAL: Can not import GUI features in a non-gui environment (DISPLAY not set)')

#=============================================================================
# Simple RFViewer
class RFViewer(GUIDataHandler):
    def __init__(self, rf_shape=None, symmetric=1, global_maximum=1, colormap='jet'):
        """
        """
        self.rf_shape = rf_shape
        self.symmetric = symmetric
        self.global_maximum = global_maximum
        self.colormap = colormap

        pyplot.ion()
        self.fig = pyplot.figure()
        pyplot.ioff()

    def append(self, tblname, value):
        value = value.T
        plot_dims = value.shape
        if len(plot_dims) == 2:
            H, D = plot_dims
            W = value.reshape( (H,)+self.rf_shape )
        elif len(plot_dims) == 3:
            H, D, C = plot_dims
            D2 = int(np.sqrt(D))
            W = value.reshape( (H, D2, D2, C) )
            # Rescaling for color plot:
            W_min = np.min(W)
            W_max = np.max(W)
            grad = 1./ (W_max - W_min)
            y_axis = W_min/ (W_min - W_max)
            W = grad * W + y_axis
        fig = self.fig
        symmetric = self.symmetric
        global_maximum = self.global_maximum

        # Set colormap
        exec('colormap = pyplot.cm.' + self.colormap)

        # Determine number of rows/cols necessary to display H RFs
        rows = int(sqrt(H))
        cols = rows
        if rows*cols < H: rows += 1
        if rows*cols < H: cols += 1
        
        # Determine plotting max and min values
        if global_maximum:
            plot_value_max = np.max(W)
            plot_value_min = np.min(W)
            if symmetric:            
                plot_value_max = max(np.abs(plot_value_max), np.abs(plot_value_min))
                plot_value_min = -plot_value_max

        # Draw RFs onto grid
        pyplot.ioff()
        pyplot.figure(self.fig.number)
        for h in xrange(H):
            if not global_maximum:
                plot_value_max = np.max(W[h])
                plot_value_min = np.min(W[h])
                if symmetric:
                    plot_value_max = max(np.abs(plot_value_max), np.abs(plot_value_min))
                    plot_value_min = -plot_value_max
            r = h // cols
            c = h % cols
            spl = fig.add_subplot(rows, cols, h+1)
            spl.cla()
            if len(plot_dims) == 2:
                spl.imshow(W[h], interpolation='nearest', cmap=colormap, vmin= plot_value_min, vmax=plot_value_max)
            if len(plot_dims) == 3:
                spl.imshow(W[h], interpolation='nearest')
            pyplot.axis('off')

        # Refresh display
        pyplot.ion()
        pyplot.draw()

#=============================================================================
# Feature Plottor

class FeaturePlotter(GUIDataHandler):
    def __init__(self):
        """ """
        self.data = {}
        
        pyplot.ion()
        self.fig = pyplot.figure()
        pyplot.ioff()
        
    def append(self, tblname, value):
        H, C = value.shape
        T = value
        
        # Rescaling for color plot:
        T_min = np.min(T)
        T_max = np.max(T)
        grad = 1./ (T_max - T_min)
        y_axis = T_min/ (T_min - T_max)
        T = grad * T + y_axis
        
        fig = self.fig
        rows = int(sqrt(H))
        cols = rows
        if rows*cols < H: rows += 1
        if rows*cols < H: cols += 1
        # Draw RFs onto grid
        pyplot.ioff()
        pyplot.figure(self.fig.number)
        for h in xrange(H):
            spl = fig.add_subplot(rows, cols, h+1)
            spl.cla()
            spl.imshow(T[None, None, h], interpolation='nearest')
            pyplot.axis('off')

        # Refresh display
        pyplot.ion()
        pyplot.draw()


#=============================================================================
# Y over T plotter

class YTPlotter(GUIDataHandler):
    def __init__(self):
        """ """
        self.data = {}

        pyplot.ion()
        self.fig = pyplot.figure()
        pyplot.ioff()

    def append(self, tblname, value):
        if tblname not in self.data:
            self.data[tblname] = []

        arr = self.data[tblname]
        arr.append(value)
            
        # Redraw everything (XXX far from optimal XXX)
        pyplot.figure(self.fig.number)
        spl = self.fig.add_subplot(1, 1, 1)
        spl.cla()

        for key, arr in self.data.items():
            spl.plot(arr)

        # Refresh display
        pyplot.ion()
        pyplot.draw()
        pyplot.ioff()


#=============================================================================
# Y over T plotter

class AudioPlotter(GUIDataHandler):
    def __init__(self):
        """ """
        pyplot.ion()
        self.fig = pyplot.figure()
        pyplot.ioff()
    
    def append(self, tblname, value):
        H, D = value.shape
        fig = self.fig
        
        # Determin number of rows/cols necessary to display H RFs
        rows = int(sqrt(H))
        cols = rows
        if rows*cols < H: rows += 1
        if rows*cols < H: cols += 1

        pyplot.figure(self.fig.number)
        for h in xrange(H):
            r = h // cols
            c = h % cols
            spl = fig.add_subplot(rows, cols, h+1)
            spl.cla()
            spl.plot(value[h])
            pyplot.axis('off')

        # Refresh display
        pyplot.ion()
        pyplot.draw()
        pyplot.ioff()

#=============================================================================
# GUI Main Programm
class GUI(object):
    def __init__(self, gui_queue):
        #print "GUI: Process started..."
        self.gui_queue = gui_queue
        self.viz_map = {}

    def run(self):
        """ Receive commands via self.gui_queue """
        viz_map = self.viz_map

        while True:
            pkt = self.gui_queue.get()
            cmd = pkt['cmd']
            vizid = pkt['vizid']
            
            if cmd == 'create': 
                handler_class = pkt['handler_class']
                handler_args  = pkt['handler_args']
                handler_kargs = pkt['handler_kargs']

                #print "GUI: Creating viz_map[%s] = %s(...)" % (vizid, handler_class)
                viz_map[vizid] = handler_class(*handler_args, **handler_kargs)   # Instantiate GUIDataHandler
            
            if cmd == 'append':
                #print "GUI: viz_map[%s].append( ... )" % vizid
                tblname = pkt['tblname']
                value = pkt['value']
                viz_map[vizid].append(tblname, value)

            if cmd == 'append_all':
                #print "GUI: viz_map[%s].append_all( ... )" % vizid
                valdict = pkt['valdict']
                viz_map[vizid].append_all(valdict)

            if cmd == 'close':
                viz_map[vizid].close()
                del viz_map[vizid]

            if cmd == 'quit':
                return


#=============================================================================
# Data Plotter for after the algorithm has run
class DataPlotter():
    """ Expects an h5file containing the Ws, Pis, and Sigmas, OR the MAEs
    """
    def __init__(self, h5file):
        self.h5file = openFile(h5file)
        self.variables = self.h5file.root.__members__
        
    def plotW(self, iteration=None):
        """ Plots a W. If no iteration is given the last W is plotted. If 

        an iteration is given, the very iteration is plotted. (Notice the 

        difference to PictureCreator here)
        """
        if 'W' in self.variables:
            W = self.h5file.root.W[:]
            num_of_W, H, D = W.shape
            if iteration >= num_of_W:
                raise ValueError("Given iteration is too large.")
            if iteration == None:
                iteration = num_of_W-1
            W = W[iteration,:,:]
        elif 'W_final' not in self.variables:
            W = self.h5file.root.W_final[:]
            H, D = W.shape
        else:
            raise ValueError("This DataPlotter doesn't contain W.")
        D2 = np.sqrt(D)
        pyplot.figure()
        rows = int(np.ceil(sqrt(H)))
        cols = rows
        for ind_plot in xrange(H):
            spl = pl.subplot(rows, cols, ind_plot+1)
            spl.cla()
            spl.imshow(W[ind_plot].reshape(D2,D2), interpolation='nearest', cmap=pyplot.cm.jet)
            pyplot.axis('off')
        pl.show()
        
    def plotMAE(self, num_of_colums=None):
        """ Creates a histogram of the MAEs. num_of_colums can be set and will then be passed to
            hist.
        """
        if 'MAE' not in self.variables:
            raise ValueError("This DataPlotter doesn't contain MAE.")
        MAE = self.h5file.root.MAE[:]
        pyplot.figure()
        if num_of_colums == None:
            pyplot.hist(MAE)
        else:
            pyplot.hist(MAE, num_of_colums)
        pl.show()
        
    def plot_scalar(self, name):
        """ Provide a name of a variable contained in the DataPlotter object. 
            This variable will then be plotted along an iterarion axis.
        """
        if name not in self.variables:
            raise ValueError("This DataPlotter doesn't contain " + name + ".")
        to_plot = '/' + name
        to_plot = self.h5file.getNode(to_plot)[:]
        pyplot.figure()
        pyplot.plot(to_plot, '.')
        
    def close(self):
        self.h5file.close()

