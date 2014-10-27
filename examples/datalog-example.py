#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#

import sys
sys.path.insert(0, "..")

from time import sleep
import numpy as np

from pulp.utils.parallel import pprint
from pulp.utils.datalog import dlog, StoreToH5, TextPrinter
from pulp.visualize.gui import GUI, RFViewer, YTPlotter

# Parameters
rf_shape = (26, 26)
H = 16

# Configure Data-Logger
dlog.start_gui(GUI)
dlog.set_handler('W', RFViewer, rf_shape=rf_shape)
dlog.set_handler('S', YTPlotter)
dlog.set_handler('C', YTPlotter)
dlog.set_handler(('T', 'S', 'C'), TextPrinter)

# And GO!
D = rf_shape[0] * rf_shape[1]

Wshape = (H,D)
i = 0
for T in np.linspace(0., 20, 50):
    i = i + 1
    pprint( "%i th iteration..." % i)

    W = np.random.normal(size=Wshape)
    dlog.append_all( {
        'T': T,
        'W': W,
        'S': np.sin(T),
        'C': np.cos(T),
    } )


dlog.close()
