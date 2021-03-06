#!/usr/bin/env python
#
#  Lincense: Academic Free License (AFL) v3.0
#
import sys

from time import sleep
import numpy as np

from prosper.utils.parallel import pprint
from prosper.utils.datalog import dlog, StoreToH5, TextPrinter

# Parameters
rf_shape = (26, 26)
H = 16

# Configure Data-Logger
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
