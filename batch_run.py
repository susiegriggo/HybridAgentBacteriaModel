"""
Run the simulation with multithreading to decrease computational time
"""

from mesa.batchrunner import BatchRunnerMP
from multiprocessing import freeze_support
from model import *

fixed_params = {
    "width": 0.1,
    "height": 0.01,
    "population":100
}

variable_params = {"name": ['batch1','batch2']}

if __name__ == "__main__":

    freeze_support()

    batch_run = BatchRunnerMP(Tube, nr_processes=None, fixed_parameters=fixed_params, variable_parameters= variable_params, iterations=1, max_steps=100, model_reporters= None, agent_reporters=None, display_progress= True)

    a = batch_run.run_all()

    #get the data
    #run_data = batch_run.get_model_vars_dataframe()
    #print(run_data)
    #print(np.min(run_data['Concentration'][0]))

    #reintroduce periodically saving the concentration field and density
    #save the final concentration field and density
