from mesa.visualization.ModularVisualization import ModularServer

from model import Tube
from SimpleContinuousModule import SimpleCanvas


def bacteria_draw(agent):
    return {"Shape": "circle", "r": 8, "Filled": "true", "Color": "Blue"}


bacteria_canvas = SimpleCanvas(bacteria_draw, 500, 500) #this controls the size of the visualised space in the server
model_params = {
    "population": 50,
    "width":0.01,
    "height": 0.01,
    "name": 'vis'
}


server = ModularServer(Tube, [bacteria_canvas], "Bacteria", model_params)