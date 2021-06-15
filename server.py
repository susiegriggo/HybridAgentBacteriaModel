from mesa.visualization.ModularVisualization import ModularServer

from model import Tube
from SimpleContinuousModule import SimpleCanvas

def bacteria_draw(agent):
    return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}

bacteria_canvas = SimpleCanvas(bacteria_draw, 500, 500) #this controls the size of the visualised space in the server

model_params = {
    "population":1000,
    "width": 20, #these parameters set the scale for the model
    "height": 1,
}


#server = ModularServer(Tube, [bacteria_canvas], "Bacteria", model_params)
model = Tube(1000,20,1) 
for i in range(10):
    model.step()
    print('step: '+str(i))
