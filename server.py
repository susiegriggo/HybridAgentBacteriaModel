"""
Run the simulation using this script.
The population size, width and height of the modelling space can be set using the command line.
"""
from model import Tube
import argparse
import time 

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--population", help = 'number of cells in the starting bacterial population', type = int)
parser.add_argument("-w", "--width", help = 'width of the modelling space', type = float)
parser.add_argument( "-he", "--height", help = 'height of the modelling space', type = float)
parser.add_argument( "-s", "--steps", help = 'number of steps for the model to perform', type = int)
parser.add_argument("-n", "--name", help = 'prefix of the output files', type = str)
parser.add_argument("-pat", "--pattern", help = 'motility pattern of the bacteria', type = str, default = 'tumble') 
parser.add_argument("-b", "--beta", help = 'scaled value for bacterial consumption', type = float, default = False) 
parser.add_argument("-c", "--c", help = 'scaled value for the starting concentration of attractant', type = float, default = False) 
parser.add_argument("-dx", "--dx", help = 'space between nodes - dx', type = float, default = False) 
parser.add_argument("-dt", "--dt", help = 'size of the timesteps - dt', type = float, default = False) 
args = parser.parse_args()

print('***STARTING SIMULATION***')

model = Tube(args.population, args.width, args.height, args.name, args.pattern, args.beta, args.c, dx_ = args.dx, dt = args.dt)

print('Simulation started using...')
print(str(args.population)+' cells')
print(str(args.pattern) +' pattern')
print('width: '+str(args.width) + ' height: '+str(args.height))

#intialise a timer 
start = time.time() 

for i in range(args.steps):

    model.step()

    if (i+1)%10 == 0:
        print('step: '+str(i+1), flush = True)

#end time 
end = time.time() 

print('***COMPLETE***')
print('Completed in '+ str(end-start)+' seconds') 
