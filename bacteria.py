"""
Creates a bacteria object which can be added to a tube model
"""
import numpy as np
import random
import pandas as pd
from mesa import Agent
import math

radius = 10E-4  # cm
D_rot = 0.062 #rational diffusion coefficient radians^2/s
D_rot = D_rot*10E-9 #change the units to cm
epsilon = 10E-16#adjusts edges of the modelling space- must be sufficiently small or causes errors with wall effects

alpha =  1  #bias of the bacteria to the nutrients it is consuming based on the previous run duration
doubling_mean = 360
doubling_std = 20
doubling_mean = 10E+26
doubling_std = 0

velocity_mean = 2.41E-3 #mean velocity in cm/s
velocity_std = 6E-4 #standrd deviation of the velocity

#wall effects
arch_collision = 0 #probability of an arch collision
tangent_collision = 1 - arch_collision #probability of a tangential deflection collision

#list of positions
pos_list =[]

class Bacteria(Agent):
    """

    Creates a bacteria agent
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
    ):
        """
        Create a new Bacteria

        Args:
            unique_id: Unique agent identifier.
            pos: Starting position
            width: size of the modelling space
            height: size fo the modelling space
            daughter: True/False of whether bacteria is a daughter cell of a cell in the model
            pattern: 'tumble' for tun and tumble, 'reverse' for run and reverse and 'reverse_flick' for run, reverse flick
        """

        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.pos = np.array(pos)
        self.model_name = model_name #name of the model which the agent resides
        self.pattern = pattern

        self.ang_mean = 68
        self.ang_std = 37
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)
        self.mean_run = 1
        self.mean_tumble = 0.1
        self.status = 0 #determines whether running or tumbling. 0 is tumbling, 1 is running, 2 is extending a run
        self.duration = self.getDuration(self.mean_tumble) #stores the time of the duration
        self.dt = 0.01 #time for each tick in the simulation
        self.timer = 0  #traces where up to on the current run/tumble
        self.ang = 0 # angle for running
        self.step_counter = 0 #count the number of steps the agent has done

        #wiener process for rotational diffusion
        self.W_x = np.sqrt(self.dt) * np.random.normal(0, 1, 1)
        self.W_y = np.sqrt(self.dt) * np.random.normal(0, 1, 1)
        self.c_start = 0 #concentration of attractant at the start of the run
        self.c_end = 0 #concentration of attractant at the end of the run

        self.next_double = np.random.normal(doubling_mean, doubling_std, 1)

        if daughter == False:
            #if the bacteria is new it could be anywhere in its growth cycle
            self.next_double = np.random.uniform(0, self.next_double,1)

        global model_width
        global model_height
        model_width = width
        model_height = height

    def getAngle(self, ang_mean, ang_std):
        """
        get the angle for the next reorientation from Bergs lognormal distribution
        with mean of 68 degrees with a standard deviation of 37 degrees

        The angle selected may in the positive or negative direction
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
        Get the duration from a Poisson distribution.
        Used to get the tumble or run duration
        """
        #generate a random number between 0 and 1
        p = random.uniform(0,1)

        #get the duration
        dur = (-1)*mean_t*np.log(p)

        return dur

    def checkCollision(self, pos):
        """
        Check if the bacteria collide with the side of the tube.
        If bacteria collide shorten the run and induce a tumble.
        A small value - episilon is added to keep the bacterium in the modelling space.
        To determine whether an arch or tagent collision occurs a value is drawn from a uniform distribution between 0 and 1
        """

        #get the current position
        x = pos[0]
        y = pos[1]

        #bacteria are hitting the left wall
        if x < 0:

            #adjust position
            x = 0+epsilon

            #determine whether an arch or tangent collision occurs
            p = random.uniform(0, 1)

            #perform tangent collision otherwise arch
            if p >= arch_collision:
                self.ang = 180-self.ang


        #bacteria are hitting the right wall
        elif x > model_width:

            #adjust position
            x = model_width - epsilon

            #determine whether an arch or tangent collision occurs
            p = random.uniform(0, 1)

            #perform tangent colision otherwise arch
            if p >= arch_collision:
                self.ang = 180-self.ang

        #bacteria are hitting the top wall
        if y < 0:

            #adjust position
            y = 0+epsilon

            #determine whether an arch or tangent collision occurs
            p = random.uniform(0, 1)

            #perform tangent collision otherwise arch
            if p >= arch_collision:
                self.ang = 360-self.ang
            #self.ang = self.ang*-1

        #bacteria are hitting the bottom wall
        elif y > model_height:

            #adjust the position
            y = model_height-epsilon

            #determine whether a tangent or arch collision occurs
            p = random.uniform(0, 1)

            #perform tangent collision otherwise arch
            if p >= arch_collision:
                self.ang = 360 - self.ang

        return [x,y]

    def getConcentration(self):
        """
        Get the concentration of attractant at a location for the currently saved concentration field
        :return:
        """

        #the the relevant csv file
        conc_file = str(self.model_name) + '_concentration_field.csv'
        conc_field = pd.read_csv(conc_file, sep = ',')
        #get the number of columns and rows in the concentration field
        conc_nx = len(conc_field.columns) - 1
        conc_ny = len(conc_field) - 1

        #get the increase in each direction per unit time
        conc_dx = model_width/conc_nx
        conc_dy = model_height/conc_ny

        #find the number of decimal places to round the position to
        round_dx = np.log10(conc_dx)*-1
        round_dy = np.log10(conc_dy)*-1

        #get the concentration corresponding to this rounded concentration
        field_pos = [int(round(self.pos[0],int(round_dx))/conc_dx), int(round(self.pos[1], int(round_dy))/conc_dy)]

        #get this position in the concentration dataframe
        current_pos = conc_field.loc[field_pos[1]][field_pos[0]]
        return current_pos

    def neighbourCollide(self):
        """
        Get the neighbours which are in the vicinity of the bacteria and see if they collide
        :return:
        """

        #get the neighbours of the bacteria
        vision = radius #vision is the radius of the bacteria
        colliders = self.model.space.get_neighbors(self.pos, vision, False)

        #if there are colliding bacteria generate new run angles
        if len(colliders) > 0:

            #generate a new run angle
            self.ang = self.getAngle(self.ang_mean, self.ang_std) + self.ang
            self.ang = self.ang % 360

            #give the colliders new angles as well
            for collider in colliders:
                collider.ang = (self.getAngle(self.ang_mean, self.ang_std) +self.ang)
                collider.ang = collider.ang %360

    def tumbleStep(self):
        """
        Adjust the timers for a run and tumble motility pattern
        :return:
        """
        if self.timer >= self.duration:

            # if currently tumbling change to run
            if self.status == 0:
                self.status = 1
                # generate a new running angle
                self.ang = self.getAngle(self.ang_mean, self.ang_std) + self.ang
                self.ang = self.ang %360
                # get the duration of the next run
                self.duration = self.getDuration(self.mean_run)
                self.velocity = np.random.normal(velocity_mean, velocity_std, 1)

            # if its not tumbling its either extending or running
            else:

                if self.status != 2:
                    # if not biasing movement already see if it can be biased
                    #current_conc = self.pos[0]
                    current_conc = self.getConcentration()
                    self.c_start = self.c_end
                    self.c_end = current_conc

                    if self.c_end > self.c_start:
                        self.duration = alpha * self.duration
                        #self.duration = alpha * self.getDuration(self.mean_run)
                        self.status = 2
        
                else:

                    # if biasing already it is time to tumble
                    self.status = 0
                    # get the duration of the tumble
                    self.duration = self.getDuration(self.mean_tumble)

            # reset timer
            self.timer = 0

   # def reverseStep(self):
        """
        Adjust timers for a run and reverse motility pattern
        :return:
        """
    #    if self.timer >= self.duration:

            #if currently running change to reverse



    def step(self):
        """
        Perform operations to move bacteria for one tick
        """

        #if bacteria have just doubled reset the doubling timer
        if self.next_double < self.dt:
            self.next_double = np.random.normal(doubling_mean, doubling_std, 1)

        #check if the bacteria is about to collide
        self.neighbourCollide()

        #get the current timestep in the run/tumble
        step_num = int(self.timer/self.dt)
        x_new = self.pos[0] + self.velocity*self.status*np.cos(np.deg2rad(self.ang))*self.dt+np.sqrt(2*D_rot*self.dt)*self.W_x
        y_new = self.pos[1] + self.velocity*self.status*np.sin(np.deg2rad(self.ang))*self.dt+np.sqrt(2*D_rot*self.dt)*self.W_y
        new_pos = [x_new[0], y_new[0]]

        #check if bacterium collide with edges of the modelling space
        new_pos = self.checkCollision(new_pos)
        self.pos = new_pos

        #move the agent
        self.model.space.move_agent(self, self.pos)

        #add this timestep to the timer
        self.timer = self.timer + self.dt
        #update the time until the next replication
        self.next_double = self.next_double - self.dt

        #update the wiener processes ready for the next tick
        self.W_x = self.W_x + np.sqrt(self.dt) * np.random.normal(0,1,1)
        self.W_y = self.W_y + np.sqrt(self.dt) * np.random.normal(0,1,1)

        #perform adjusts necessary for the current motility pattern
        if self.pattern == 'tumble':
            self.tumbleStep()

    """
	#uncomment only to make the figure of the run and tumble motlity pattern 
	#save the positions to a file
        self.step_counter = self.step_counter + 1
        pos_list.append(self.pos)

        #save only every 10 second
        if self.step_counter % 1000 == 0:
            pos_df = pd.DataFrame({'position': pos_list})
            pos_df .to_csv('example_position_list14.csv', index = False)

        """

        #hdebugging lines which can be uncommented
        #print('run duration: '+str(self.duration))
        #print('run timer: '+str(self.timer))
        #print('status: '+str(self.status))
        #print('angle: '+str(self.ang))
        #print('velocity: '+str(self.velocity))
        #print('')






