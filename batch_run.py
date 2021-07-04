"""
Run the simulation with multithreading to decrease computational time
"""

from mesa.batchrunner import BatchRunnerMP
from multiprocessing import freeze_support
from model import *

fixed_params = {
    "width": 1,
    "height": 0.1,
    "population": 10
}

variable_params = {"name": ["batch1", "batch2"]}

if __name__ == "__main__":

    freeze_support()

    batch_run = BatchRunnerMP(Tube, nr_processes=None, fixed_parameters=fixed_params, variable_parameters= variable_params, iterations=1, max_steps=1000, model_reporters= {"Density": compute_density, "Concentration": compute_concentration}, agent_reporters=None, display_progress= True)

    batch_run.run_all()

    ##get the data from the final model
    #data_collector_model = batch_run.get_collector_model()
    ##show the dictionary
    #print(data_collector_model)

    run_data = batch_run.get_model_vars_dataframe()
    print(run_data)