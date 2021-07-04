"""
Run the simulation with multithreading to decrease computational time
"""

from mesa.batchrunner import BatchRunnerMP
from multiprocessing import freeze_support
from model import Tube

fixed_params = {
    "width": 1,
    "height": 1,
}

variable_params = {"population": range(10,12)}

if __name__ == "__main__":

    freeze_support()

    batch_run = BatchRunnerMP(Tube, nr_processes=None, fixed_parameters=fixed_params, variable_parameters= variable_params, iterations=2, max_steps=100, model_reporters= None, agent_reporters=None, display_progress= True)

    batch_run.run_all()