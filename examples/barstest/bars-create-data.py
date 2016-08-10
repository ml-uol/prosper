#!/usr/bin/env python

from __future__ import division

import sys
sys.path.insert(0, '../..')

import numpy as np

from pulp.utils import create_output_path 
from pulp.utils.autotable import AutoTable

#
if "param_file" not in globals():
    param_file = sys.argv[1]

if "output_path" not in globals():
    output_path = sys.argv[2]

#=============================================================================
# Read parameter file
params = {} 
execfile(param_file, params)

#=============================================================================
# Some sanity checks
H = params['H']
D = params['D']
N = params['N']   

D2 = H // 2   # size of pixel image: D2 \times D2

assert D == D2**2

#=============================================================================
# Use model to generate data 
model     = params['model']
params_gt = params['params_gt']

data = model.generate_data(params_gt, N)

# and save results
out_fname = output_path+"/data.h5"
with AutoTable(out_fname) as tbl:
    # Save ground-truth parameters
    for key in params_gt:
        tbl.append(key, params_gt[key])

    # Save generated data
    for key in data:
        tbl.appendList(key, data[key])
