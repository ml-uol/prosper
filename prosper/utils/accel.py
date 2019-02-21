#
#  Lincense: Academic Free License (AFL) v3.0
#
"""

Try to import accelerated versions of various NumPy operations..

"""

import numpy as np

#=============================================================================
# Pull default implementations

backend = "numpy"

sin = np.sin
cos = np.cos
exp = np.exp
log = np.log
log2 = np.log2
log10 = np.log10

#=============================================================================
# Try to import AMD Core Math Library
try:
    import pyacml

    backend = "pyacml"

    sin = pyacml.sin
    cos = pyacml.cos
    exp = pyacml.exp
    log = pyacml.log
    log2 = pyacml.log2
    log10 = pyacml.log10
except ImportError as e:
    pass
