"""
Run the simulation using this script.
The population size, width and height of the modelling space can be set using the command line.
"""
from model import Tube
import argparse
import time 


#default values 

#variables to control the doubling time of the bacteria (seconds)
doubling_mean = 360
doubling_std = 20

#variables to control the veolicty of the bacteria (cm/s)
velocity_mean = 2.41E-3
velocity_std = 6E-4

#set the mean duration of a run in a model
mean_run = 1

#defualt run distribution 
dist = 'poisson'

#set the default tumble angle and standard deviation
tumble_angle_mean = 68
tumble_angle_std = 37
 
#arguments to parse
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
parser.add_argument("-vm", "--velocity_mean", help = 'the mean veloicty for a run', type = float, default = velocity_mean) 
parser.add_argument("-vs", "--velocity_std", help = 'the standard deviation of the velocity for a run', type = float, default = velocity_std)
parser.add_argument("-dm", "--doubling_mean", help = 'the mean doubling time for bacterial reproduction', type = float, default = doubling_mean) 
parser.add_argument("-ds", "--doubling_std", help = 'the standard deviation of the doubling time for bacterial reproduction', type = float, default = doubling_std)
parser.add_argument("-mr", "--mean_run", help = 'the mean run time for a bacterial run', type = float, default = mean_run)
parser.add_argument("-rd", "--run_distribution", help = 'the distribution used to draw the run and tumble durations', type = str, default = dist) 
parser.add_argument("-tumb_mean", "--tumble_angle_mean", help = 'the mean tumble angle to use for the tumble reorientations', type = float, default = tumble_angle_mean) 
parser.add_argument("-tumb_std", "--tumble_angle_std", help = "the standard deviation of tumble angle to use for the tumble reorientations", type = float, default = tumble_angle_std) 
parser.add_argument("-dt", "--dt", help = 'size of the timesteps - dt', type = float, default = 0.01) 
args = parser.parse_args()

print('***STARTING SIMULATION***')

model = Tube(args.population,
        args.width,
        args.height, 
        args.name, 
        args.pattern, 
        args.beta, 
        args.c, 
        velocity_mean = args.velocity_mean, 
        velocity_std = args.velocity_std, 
        doubling_mean =  args.doubling_mean, 
        doubling_std = args.doubling_std, 
        mean_run = args.mean_run, 
        run_dist = args.run_distribution, 
        tumble_angle_mean= args.tumble_angle_mean, 
        tumble_angle_std = args.tumble_angle_std, 
        dx_ = args.dx,  
        dt = args.dt)

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
