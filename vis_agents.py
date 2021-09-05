
from mesa.visualization.ModularVisualization import ModularServer

from model import Tube
from SimpleContinuousModule import SimpleCanvas


def bacteria_draw(agent):
    return {"Shape": "circle", "r": 8, "Filled": "true", "Color": "Blue"}


bacteria_canvas = SimpleCanvas(bacteria_draw, 500, 500) #this controls the size of the visualised space in the server
model_params = {
    "population": 10000,
    "width_":0.01,
    "height_": 0.001,
    "name": 'all_biased_concx_fixedbias',
    "pattern": 'tumble',
    "beta_star": 1E-12,
    "dx_": 0.00001,
    "dt": 0.03,
    "velocity_std": 0,
    "run_dist": "uniform",
    "tumble_angle_std":0,


}
server = ModularServer(Tube, [bacteria_canvas], "Bacteria", model_params)