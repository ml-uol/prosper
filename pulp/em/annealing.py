#
#  Lincense: Academic Free License (AFL) v3.0
#

"""

"""

from __future__ import division
try:
    from abc import ABCMeta, abstractmethod
except ImportError, e:
    from pulp.utils.py25_compatibility import _py25_ABCMeta as ABCMeta
    from pulp.utils.py25_compatibility import _py25_abstractmethod as abstractmethod

import numpy as np
import time
from pulp.utils.datalog import dlog

class Annealing():
    """ Base class for implementations of annealing schemes

    Implementations deriving from this class control the cooling schedule 
    and provide some additional control functions used in the EM algorithm.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        """ Reset the cooling-cycle. This call returs the initial cooling temperature that will be used
            for the first step.
        """
        return True;

    @abstractmethod
    def next(self, gain):
        """ Returns a (accept, T, finished)-tuple. 

            *accept* is a boolean and indicates if the parameters changed by *gain* last iteration, 
               EM should accept the new parameters or if it should bae the next iteration on 
               the old ones.

            *finished* is also a boolean and indicate whether the cooling has finished and EM should 
               drop out of the loop.

            *T* is the temperature EM should use in the next iteration
        """
        pass;


class LinearAnnealing(Annealing):
    """
    """
    def __init__(self,  steps=80):
        """
        """
        self.steps = steps
        self.anneal_params = {}
        self.reset()
        self['max_step'] = [(steps, steps)]
        self['position'] = [(0, 0.), (steps, 1.)]
        self['step'] = [(0,0.), (steps, steps)]
        self.crit_params = []
        
    def add_param(self, param_name, points):
        if np.isscalar(points):
            points = [(0,points)]

        points_to_store = []
        for point in points:
            if not isinstance(point, tuple):
                raise TypeError("points must be a list of (pos, val)-tuples")
            pos, val = point

            if isinstance(pos, float):
                pos = int(pos*self.steps)
            if pos < 0:
                pos = self.steps + pos
            
            points_to_store.append( (pos, val) )
        
        first_pos, first_val = points_to_store[0]
        if first_pos != 0:
            points_to_store.insert(0, (0, first_val))

        last_pos, last_val = points_to_store[-1]
        if last_pos != self.steps:
            points_to_store.append((self.steps+1, last_val))
        
        self.anneal_params[param_name] = points_to_store

    def __getitem__(self, param_name):
        cur_pos = self.cur_pos

        if not param_name in self.anneal_params:
            return 0.0

        points = self.anneal_params[param_name]

        for i in xrange(len(points)):
            pos, _ = points[i]
            if pos > cur_pos:
                break

        left_pos, left_val = points[i-1]
        right_pos, right_val = points[i]
            
        frac = (cur_pos-left_pos) / (right_pos-left_pos)
        return frac * (right_val-left_val) + left_val

    def __setitem__(self, param_name, points):
        self.add_param(param_name, points)

    def reset(self):
        self.cur_pos = 0
        self.finished = False

    def next(self, gain=0.0):
        """
        Step forward by one step.

        After calling this method, this annealing object will
        potentially return different values for all its values.
        """
        if self.finished:
            raise RuntimeError("Should not next() further when already finished!")

        self.accept = True
        self.cur_pos = self.cur_pos + 1
        
        if self.cur_pos >= self.steps:
            self.finished = True

    def as_dict(self):
        """
        Return all annealing parameters with their current value as dict.
        """
        d = {}
        for param_name in self.anneal_params:
            d[param_name] = self[param_name]
        return d


class ConvAnnealing(Annealing):
    '''
    This Annealing scheme will iterate EM algorithm until convergence at each temperature.
    '''
    
    def __init__(self, config):
        '''
        Initialize from a configuration (dict).
        The configuration should contain:
        
        '''
        Annealing.__init__(self)
        self.useAnnealing = config['use Annealing'] #whether to use annealing
        if self.useAnnealing:
            self.startTemp = config['start Temperature'] # the starting temperature
            self.steps = config['Annealing steps'] # the number of annealing steps
            self.stepsize = (self.startTemp - 1.0)/self.steps
            self.saconv = config.get('SA Convergence', self.conv) # the judgement of convergence during annealing
        if self.useAnnealing: # The current Temperature
            self.curTemp = self.startTemp
        else:
            self.curTemp = 1.0
        self.conv = config['Convergence'] #the judgement of convergence
        self.itr = 0
        self.start_time = time.clock()
        self.accept = True
        self.finished = False
        
    def reset(self):
        '''
        Reset the temperature to the initial value
        '''
        if self.useAnnealing:
            self.curTemp = self.startTemp
        else:
            self.curTemp = 1.0
        self.accept = True
        self.finished = False
        self.itr = 0
        self.start_time = time.clock()
        return None
    
    def next(self, gain):
        if self.curTemp != 1.0:
            if abs(gain)<self.saconv:
                self.curTemp = self.curTemp - self.stepsize
                if self.curTemp <1.0:
                    self.curTemp = 1.0
        else:
            if abs(gain)<self.conv:
                self.finished = True
        self.itr+=1
        
    def getElapsedTime(self):
        return time.clock() - self.start_time

    
