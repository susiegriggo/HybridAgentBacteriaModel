"""
Run the simulation with multithreading to decrease computational time
"""

from mesa.batchrunner import BatchRunner
from multiprocessing import freeze_support
from model import *

fixed_params = {
    "width_": 0.1,
    "height_": 0.1,
}

 
#vary the variable name to introduce different replicates 

#variable_params = {"name": ['001cm_plate/001cm'], "population": [1000, 10000, 100000], "c_star":[0.1, 1], "beta_star": [10E-6,10E-7, 10E-8, 10E-9], "pattern":['tumble', 'reverse', 'flick'], 'dt':[0.01, 0.03, 0.05] , 'dx_':[0.0001] }
variable_params = {"name": ['mini_tube_test/tube'], "population": [10000, 1000], "c_star":[ 1], "beta_star": [ 10E-9, 10E-10, 10E-11, 10E-12], "pattern":['tumble'], 'dt':[0.01] , 'dx_':[0.0001] }


if __name__ == "__main__":

    freeze_support()

    batch_run = BatchRunnerMP(Tube,  fixed_parameters=fixed_params, nr_processes = None, variable_parameters= variable_params, iterations=1, max_steps=300000, model_reporters= None, agent_reporters=None, display_progress= True)

    a = batch_run.run_all()
