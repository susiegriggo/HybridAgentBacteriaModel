"""
Run the simulation using this script.
The population size, width and height of the modelling space can be set using the command line.
"""
from model import Tube
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--population", help = 'number of cells in the starting bacterial population', type = int)
parser.add_argument("-w", "--width", help = 'width of the modelling space', type = float)
parser.add_argument( "-he", "--height", help = 'height of the modelling space', type = float)
parser.add_argument( "-s", "--steps", help = 'number of steps for the model to perform', type = int)
parser.add_argument("-n", "--name", help = 'prefix of the output files', type = str)
args = parser.parse_args()

print('STARTING SIMULATION')

model = Tube(args.population, args.width, args.height, args.name)
for i in range(args.steps):
    model.step()
    print('step: '+str(i+1), flush = True)

#print a summary
print('COMPLETE')
print('Starting number of bacteria: '+str(args.population))
