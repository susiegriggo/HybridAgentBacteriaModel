"""
Run the simulation with multithreading to decrease computational time
"""

from mesa.batchrunner import BatchRunnerMP
from multiprocessing import freeze_support
from model import *

fixed_params = {
    "width": 1,
    "height": 0.1,
}


#vary a range of different parameters 
#vary the variable name to introduce different replicates 
variable_params = {"name": ['rep1', 'rep2', 'rep3'], "population": [10E-4, 10E-6, 10E-8], "c_star":[0.01, 0.1, 1], "beta_star": [10E-8,10E-6, 10E-4]}

if __name__ == "__main__":

    freeze_support()

    batch_run = BatchRunnerMP(Tube, nr_processes=None, fixed_parameters=fixed_params, variable_parameters= variable_params, iterations=1, max_steps=100000, model_reporters= None, agent_reporters=None, display_progress= True)

    a = batch_run.run_all()
