"""
Creates a bacteria object which can be aded to a tube model
"""

import numpy as np
import random
import pandas as pd
from mesa import Agent
import time 

#variables to calculate the rotational diffusion coefficient 
k = 1.3807E-16 #boltzmans constant 
T = 305 #temperature in kelvin  
eta = 0.027 #viscosity in (cm sec)^-1
radius = 1E-4  # cm

#calcualte the rotational diffusion coefficient 
D_rot = (k*T)/(8*np.pi*eta*np.power(radius,3))

#small value to adjust edges of the modelling space 
epsilon = 10E-16 

#bias of the bacteria to the nutrients it is consuming 
alpha =  0.5  

#variables to control the doubling time of the bacteria (seconds) 
doubling_mean = 360
doubling_std = 20

#variables to control the veolicty of the bacteria (cm/s) 
velocity_mean = 2.41E-3 
velocity_std = 6E-4 

#set the mean duration of a run in a model 
mean_run = 1 

#wall effects
arch_collision = 1  #manually set the probability of an arch collision 
tangent_collision = 1 - arch_collision #probability of a tangential deflection collision

class Bacteria(Agent):
    """
    Create a bacterium object 
    """

    def __init__(

    self,
        unique_id,
        model,
        pos,
        width,
        height,
        daughter,
        model_name,
        pattern,
        dx, 
        velocity_mean = velocity_mean,
        velocity_std = velocity_std, 
        doubling_mean = doubling_mean, 
        doubling_std = doubling_std,
        dt = 0.01  
    ):
        """
        Create a new Bacteria agent

        Args:
            unique_id: Unique agent identifier.
            pos: Starting position
            width: size of the modelling space
            height: size fo the modelling space
            daughter: True/False of whether bacteria is a daughter cell of a cell in the model
            model_name: the prefix of the model files 
            pattern: motility pattern of the bacteria - one of tumble, reverse or flick 
            velocity_mean = mean velocity of a bacterial run 
            velocity_std = standard deviation of velocity of a bacterial run 
            doubling_mean = mean of the doubling time (seconds) of a bacterial cell 
            doubling_std = standard deviation of the doubling time (seconds) of a bacterial cell 
            dt = timestep for the simulation (s) 
        """

        super().__init__(unique_id, model)

        #throw exception if the wrong types of values are passed
        pattern_type = ['tumble', 'reverse', 'flick'] #possible movement patterns
        if pattern not in pattern_type:
            raise ValueError("Pattern must be one of tumble, reverse or flick")

        self.unique_id = unique_id
        self.pos = np.array(pos)
        self.model_name = model_name
        self.pattern = pattern
        self.doubling_mean = doubling_mean
        self.doubling_std = doubling_std
        self.velocity_mean = velocity_mean 
        self.velocity_std = velocity_std
        self.model = model
        self.dx = dx        
        self.dt = dt  

        #parameters universal to all motility patterns
        self.velocity = np.random.normal(self.velocity_mean, self.velocity_std, 1)
        self.mean_run = 1

        #set parameters for the run and tumble motility pattern 
        if self.pattern == 'tumble':
            self.ang_mean = 68
            self.ang_std = 37
            self.mean_tumble = 0.1
            self.duration = self.getDuration(self.mean_tumble)  # stores the time of the duration
            self.status = 0  #starting status, tumble = 0 , run = 1

        #set parameters for the run and reverse motility pattern 
        if self.pattern == 'reverse':
            self.reverse_std = 20  # standard deviation of angle
            self.status = 1  # 1 is running, 2 is extending a run
            self.duration = self.getDuration(self.mean_run)  # duration of the first run

        #set parameters for the run and reverse and flick motility pattern 
        if self.pattern == 'flick':
            self.status = 1  # 1 is running, 2 is extending a run, 3 is flicking
            self.flick_std = 10  # standard deviation of a flick
            self.reverse_std = 2  # standard deviation of a reverse
            self.duration = self.getDuration(self.mean_run)  # duration of the first run

        #intialise a timer to count the time elapsed for the agent 
        self.timer = 0

        #generate a random angle for the cell so that all cells do not start with the same direction 
        self.ang = np.random.uniform(0, 359, 1) 

        #store the positions the cell is at each tick 
        self.pos_list = []
        self.ticks = 1

        #wiener process for rotational diffusion
        self.W_x = np.sqrt(self.dt) * np.random.normal(0, 1, 1)
        self.W_y = np.sqrt(self.dt) * np.random.normal(0, 1, 1)

        #concentration of attractant at the start/end of each run
        self.c_start = 0
        self.c_end = 0

        #set a timer for when then cell will next reproduce
        self.next_double = np.random.normal(doubling_mean, doubling_std, 1)[0]
        #if the bacteria is not a daughter cell it could be anywhere in its growth cycle 
        if daughter == False:

           #generate a random time for its next replication 
            self.next_double = np.random.uniform(0, self.next_double,1)

        #if the the cell is a daughter cell draw a doubling time from a normal distribution  
        else: 
            self.next_double = np.random.normal(doubling_mean, doubling_std, 1)
    
        #set global variables which are reused throughout 
        global model_width
        global model_height
        model_width = width
        model_height = height

    def getTumbleAngle(self, ang_mean, ang_std):
        """
        Get the angle for the next reorientation from Bergs lognormal distribution
        with mean of 68 degrees with a standard deviation of 37 degrees
        """

        # generate two random numbers between 0 and 1
        a = np.random.rand(1)[0]
        b = np.random.rand(1)[0]

        # calculate Box-Muller transformation
        r = np.sqrt(-2 * np.log(a)) * np.cos(2 * np.pi * b)
        A = np.log((ang_mean * ang_mean) / np.sqrt((ang_mean * ang_std) + (ang_mean * ang_mean)))
        B = np.sqrt(np.log(1 + (ang_std * ang_std) / (ang_mean * ang_mean)))

        # get the tumble angle - the angle is cumulative
        angle = np.exp(B * r + A)

        # Bernoulli distribution to select direction
        d = random.randint(0, 1)
        if d == 0:
            angle = angle * (-1)

        return angle

    def getDuration(self, mean_t):
        """
        Get the duration of a run or tumble using a Poisson distribution
        """

        #generate a random number between 0 and 1
        p = random.uniform(0,1)

        #get the duration
        dur = (-1)*mean_t*np.log(p)

        return dur

    def checkCollision(self, pos):
        """
        Check if the bacteria collide with the side of the tube.
        Peform either an arching or tangental collision 
        """

        #get the coordinates of the position 
        x = pos[0]
        y = pos[1]

        #bacteria are hitting the left wall
        if x < self.dx:

            #set the cell to be at the left wall 
            x = self.dx

            #set variable to determine the type of collision
            p = random.uniform(0, 1)

            #perform an arch collision
            if p > arch_collision:
                if self.ang > 180:
                    self.ag = (self.ang + 90) % 360
                else:
                    self.ang = (self.ang - 90) % 360

        #bacteria are hitting the right wall
        elif x > model_width-self.dx:

            #set the cell to be at the right wall 
            x = model_width-self.dx-epsilon

            #set varible to dermine the type of collision
            p = random.uniform(0, 1)

            #perform an arch collision 
            if p > arch_collision:
                if self.ang > 180:
                    self.ang = (self.ang - 90) % 360
                else:
                    self.ang = (self.ang + 90) % 360

        #bacteria are hitting the bottom wall
        if y < self.dx:

            #set the cell to be at the bottom wall 
            y = self.dx

            #set variable to determine the type of collision
            p = random.uniform(0, 1)
            
            #perform an arch collision
            if p > arch_collision:
                if self.ang > 270:
                    self.ang = (self.ang + 90)% 360
                else:
                    self.ang = (self.ang - 90)%360

        #bacteria are hitting the top wall
        elif y > model_height-self.dx:

            #set the cell to be at the top wall 
            y = model_height-self.dx-epsilon

            #set variable to determine the type of collision
            p = random.uniform(0, 1)

            #perform an arch collision
            if p > arch_collision:
                if self.ang < 90:
                    self.ang = (self.ang - 90) % 360
                else:
                    self.ang = (self.ang + 90) % 360

        return [x,y]

    def getConcentration(self):
        """
        Get the concentration of attractant at a location for the currently saved concentration field
        """

        # get the increase in each direction per unit time
        conc_dx = model_width / self.model.nx 
        conc_dy = model_height / self.model.ny

        # find the number of decimal places to round the position to
        round_dx = np.log10(conc_dx) * -1
        round_dy = np.log10(conc_dy) * -1

        # get the concentration corresponding to this rounded concetration
        field_pos = [int(round(self.pos[0], int(round_dx)) / conc_dx), int(round(self.pos[1], int(round_dy)) / conc_dy)]

        #get this position in the concentration dataframe
        current_pos = self.model.u[field_pos[0], field_pos[1]]
        return current_pos

    def tumbleStep(self):
        """
        Evaluate whether the tumble step needs updating 
        """

        #check whether the duration is up
        if self.timer >= self.duration:

            #if currently tumbling 
            if self.status == 0:

                #change to a run 
                self.status = 1

                #generate a new running angle
                self.ang = (self.getTumbleAngle(self.ang_mean, self.ang_std) + self.ang) % 360
                
                #get the duration of the next run
                self.duration = self.getDuration(self.mean_run)
                
                #get the velocity of the next run 
                self.velocity = np.random.normal(self.velocity_mean, self.velocity_std, 1)

            #if currently running 
            else:
                
                #if not currently extending the run  
                if self.status != 2:

                    #compare the concentration at the start vs end of the tun 
                    current_conc = self.getConcentration()
                    self.c_start = self.c_end
                    self.c_end = current_conc
                    
                    #if the concentration has increased 
                    if self.c_end > self.c_start:

                        #extend the run
                        self.duration = alpha * self.getDuration(self.mean_run)
                        self.status = 2
                
                #if extending the run 
                else:

                    #switch to a tumble
                    self.status = 0

                    #get the duration of the tumble 
                    self.duration = self.getDuration(self.mean_tumble)

            # reset timer
            self.timer = 0

    def reverseStep(self):
        """
        Adjust timers for a run and reverse motility pattern.
        1 - running
        2 - extending the run
        """
        
        #if the run duration is up 
        if self.timer >= self.duration:

            #reset the timer
            self.timer = 0

            #if the run is being extended 
            if self.status == 2:

                #perform a reverse
                self.reverseRun()
                self.status = 1

            #if currently running
            elif self.status == 1:

                #see if the concentration has increased from the start of the run 
                # current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc

                #if the concentration has increase 
                if self.c_end > self.c_start:
    
                    #extend the run 
                    self.duration = alpha * self.duration
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 2

                #if the concentration has decreased 
                else:
    
                    #reverse the run  
                    self.reverseRun()

    def reverseFlickStep(self):
        """
        Adjust timer for a run and reverse motility pattern
            1 - flicking
            2 - reversing
            3 - extending the reverse
        """
        
        #if the run duration is up 
        if self.timer >= self.duration:

            #reset the timer
            self.timer = 0

            #if currently performing a flick
            if self.status == 1:

                #change to a reverse
                self.reverseRun()
                self.status = 2
    
            #if currently extending a reverse
            elif self.status == 3:

                #change to a flick 
                self.flick()
                self.status = 1

            #if currently reversing 
            elif self.status == 2:

                #get the concentration at the start and end of the run 
                # current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc

                #if the concentration has increased
                if self.c_end > self.c_start:

                    #extend the run 
                    self.duration = alpha * self.duration
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 3
    
                #if the concentration hasn't increased
                else:

                    #change to a flick 
                    self.flick()
                    self.status = 1

    def reverseRun(self):
        """Generates a reverse run for the run and reverse and the run and reverse and flick motility pattern"""

        # get a new angle
        reverse_ang = np.random.normal(180, self.reverse_std, 1)
        reverse_direction = random.randint(0, 1)
        self.ang = (self.ang + reverse_ang * reverse_direction) % 360

        # get a duration for the new run
        self.duration = self.getDuration(mean_run)

        # get the velocity of the new run
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)

    def flick(self):
        """Generates a flick for the run reverse flick motility pattern"""

        # get a new angle
        flick_ang = np.random.normal(90, self.flick_std, 1)

        # get a duration for the flick
        self.duration = self.getDuration(mean_run)

        # get the velocity of the new run
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)

    def step(self):
        """
        Move the bacteria accordingly
        """

        #if bacteria have just doubled reset the doubling timer
        if self.next_double < self.dt:
            self.next_double = np.random.normal(self.doubling_mean, self.doubling_std, 1)

        #get the current timestep in the run/tumble
        x_new = self.pos[0] + self.velocity*self.status*np.cos(np.deg2rad(self.ang))*self.dt#+np.sqrt(2*D_rot*self.dt)*self.W_x
        y_new = self.pos[1] + self.velocity*self.status*np.sin(np.deg2rad(self.ang))*self.dt#+np.sqrt(2*D_rot*self.dt)*self.W_y
        new_pos = [x_new[0], y_new[0]]

        #adjust the position for a collisioni 
        new_pos = self.checkCollision(new_pos)
        self.pos = new_pos
        self.model.space.move_agent(self, self.pos) 

        #add this timestep to the timer
        self.timer = self.timer + self.dt
        self.ticks = self.ticks + 1

        #update the time until the next double
        self.next_double = self.next_double - self.dt

        #update the wiener processes for rotational diffusion
        self.W_x = self.W_x + np.sqrt(self.dt) * np.random.normal(0,1,1)
        self.W_y = self.W_y + np.sqrt(self.dt) * np.random.normal(0,1,1)

        # check if the status of the cell needs to be changed
        if self.pattern == 'tumble':
            self.tumbleStep()
        if self.pattern == 'reverse':
            self.reverseStep()
        if self.pattern == 'flick':
            self.reverseFlickStep()

"""
        #save the motility pattern of one of the cells 
        if self.unique_id == 1:
            self.pos_list.append(self.pos)

            #save only every 10 second
            if self.ticks % 1000 == 0:
                pos_df = pd.DataFrame({'position': pos_list})
                pos_df .to_csv('example_position_list_tumble.csv', index = False)
"""









