#!/usr/bin/env python

import sys
sys.path.insert(0, '../..')

from mpi4py import MPI
import numpy as np
np.random.RandomState(2102)

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint
from pulp.utils.barstest import generate_bars
from pulp.utils.autotable import AutoTable
import tables
from pulp.utils.parallel import stride_data

from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing

from pulp.em.camodels.bsc_et import BSC_ET

#creates MPI safe output path
output_path = create_output_path()

# Number of datapoints to generate
N = 1000

# Each datapoint is of D = size*size
size = 5

# Diemnsionality of the model
H = 2 * size     # number of latents
D = size ** 2    # dimensionality of observed data

# Approximation parameters for Expectation Truncation
Hprime = 6
gamma = 5

#Instantiate the BSC model
model = BSC_ET(D, H, Hprime, gamma, to_learn=['W','sigma','pi'])


# Model parameters used when artificially generating 
# ground-truth data. This will NOT be used for the learning
# process.
params_gt = {
    'W'     :  10*generate_bars(H),   # this function is in bars-create-data
    'pi'    :  2. / H,
    'sigma' :  1.0
}

#The root node generates the data 
if MPI.COMM_WORLD.rank == 0:
    # create data
    data = model.generate_data(params_gt, N)

    # and save results
    out_fname = output_path + "/data.h5"
    with AutoTable(out_fname) as tbl:

        # Save ground-truth parameters
        for key in params_gt:
            tbl.append(key, params_gt[key])

        # Save generated data
        for key in data:
            tbl.append(key, data[key])
        tbl.close()


if __name__ == "__main__":
	comm = MPI.COMM_WORLD
	#synchronize processes
	comm.Barrier()
	pprint("=" * 40)
	pprint(" Running %d parallel processes" % comm.size)
	pprint("=" * 40)

	#load data on all nodes
	data_fname = output_path + "/data.h5"
	my_y=None
	with tables.openFile(data_fname, 'r') as data_h5:
		N_file = data_h5.root.y.shape[1]
		if N_file < N:
			dlog.progress(
			    "WARNING: N={} chosen but only {} data points available. ".format(N, N_file))
			N = N_file
	    # points to the part of the file that the process should read
		first_y, last_y = stride_data(N)
		my_y = data_h5.root.y[0][first_y:last_y]
		data_h5.close()
	#Put data in dictionary container
	my_data = {
	    'y': my_y,
	}

	#setting up logging/output
	print_list = ('T', 'Q', 'pi', 'sigma', 'N', 'MAE', 'L')
	dlog.set_handler(print_list, TextPrinter)
	h5store_list = ('W', 'pi', 'sigma', 'y', 'MAE', 'N','L','Q')
	dlog.set_handler(h5store_list, StoreToH5, output_path +'/result.h5')

	###### Initialize model #######
	# Initialize (Random) model parameters
	model_params = model.standard_init(my_data)
	for param_name in model_params.keys():
	    if param_name not in model.to_learn:
	        model_params[param_name]=params_gt[param_name]


	#### Choose annealing schedule #####
	#Linear Annealing
	anneal = LinearAnnealing(100)
	#Increases variance by a muliplicative factor that slowly goes down to 1
	anneal['T'] = [(0., 6.), (.3, 1.)]      # [(iteration, value),... ]
	#Reduces truncation rate so as not to prematurely exclude data 
	anneal['Ncut_factor'] = [(0, 2.0), (.25, 1.)]     
	#Simulated annealing of parameters
	anneal['W_noise'] = [(0, 2.0), (.3, 0.0)]
	#Include prior parameters in the annealing schedule
	anneal['anneal_prior'] = False


	# Create and start EM annealing
	em = EM(model=model, anneal=anneal)
	em.data = my_data
	em.lparams = model_params
	em.run()

	dlog.close(True)
	pprint("Done")