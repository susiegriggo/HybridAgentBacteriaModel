from mesa.visualization.ModularVisualization import ModularServer

from model import Tube
from SimpleContinuousModule import SimpleCanvas


def bacteria_draw(agent):
    return {"Shape": "circle", "r": 8, "Filled": "true", "Color": "Blue"}


bacteria_canvas = SimpleCanvas(bacteria_draw, 500, 500) #this controls the size of the visualised space in the server
model_params = {
    "population": 1000,
    "width":2,
    "height": 0.1,
}


server = ModularServer(Tube, [bacteria_canvas], "Bacteria", model_params)