"""
Creates a bacteria object which can be added to a tube model
"""
import numpy as np
import random
import pandas as pd
from mesa import Agent

radius = 10E-4  # cm
D_rot = 0.062*10E-9 #change the units to cm
epsilon = 10E-16#adjusts edges of the modelling space- must be sufficiently small or causes errors with wall effects

alpha =  2  #bias of the bacteria to the nutrients it is consuming based on the previous run duration
doubling_mean = 360
doubling_std = 20

velocity_mean = 2.41E-3 #mean velocity in cm/s
velocity_std = 6E-4 #standrd deviation of the velocity
mean_run = 1 #mean duration of a run

#wall effects
arch_collision = 1 #probability of an arch collision
tangent_collision = 1 - arch_collision #probability of a tangential deflection collision

#list of positions used to create figures of motility patterns
pos_list =[]

class Bacteria(Agent):
    """
    Creates a bacteria agent
    """

    def __init__(self, unique_id, model, pos, width, height, daughter, model_name, pattern):
        """
        Create a new Bacteria

        Args:
            unique_id: Unique agent identifier.
            model: which model the agent is placed in
            pos: Starting position
            width: size of the modelling space
            height: size fo the modelling space
            daughter: True/False of whether bacteria is a daughter cell of a cell in the model
            pattern: 'tumble' for tun and tumble, 'reverse' for run and reverse and 'reverse_flick' for run, reverse flick
        """

        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.pos = np.array(pos)
        self.model_name = model_name
        self.pattern = pattern


        if self.pattern == "tumble":
            self.ang_mean = 68 #mean angle
            self.ang_std = 37 #standrd deviation of angle
            self.status = 0  # 0 is tumbling, 1 is running, 2 is extending a run
            self.mean_tumble = 0.1 #duration of a tumble event
            self.duration = self.getDuration(self.mean_tumble) #duration of the first run

        if self.pattern == "reverse":
            self.reverse_std = 20 #standard deviation of angle
            self.status = 1 #1 is running, 2 is extending a run
            self.duration = self.getDuration(mean_run) #duration of the first run

        if self.pattern == "flick":
            self.status = 1 #1 is running, 2 is extending a run, 3 is flicking
            self.flick_std = 10 #standrd deviation of a flick
            self.reverse_std = 2 #standrd deviation of a reverse
            self.duration = self.getDuration(mean_run) #duration of the first run

        self.ang = np.random.uniform(0, 360, 1)  # an intial angle such that all of the angles are not synced up
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)
        self.dt = 0.01 #time for each tick in the simulation
        self.timer = 0  #traces where up to on the current run/tumble
        self.step_counter = 0 #count the number of steps the agent has done

        #wiener process for rotational diffusion
        self.W_x = np.sqrt(self.dt) * np.random.normal(0, 1, 1)
        self.W_y = np.sqrt(self.dt) * np.random.normal(0, 1, 1)

        #concentration of attractant at the start/end of each run
        self.c_start = 0
        self.c_end = 0

        #control when the bacteria will reproduce
        self.next_double = np.random.normal(doubling_mean, doubling_std, 1)
        if daughter == False:
            #if the bacteria is new it could be anywhere in its growth cycle
            self.next_double = np.random.uniform(0, self.next_double,1)

        global model_width
        global model_height
        model_width = width
        model_height = height

    def getTumbleAngle(self, ang_mean, ang_std):
        """
        Get the angle for the next reorientation from Bergs lognormal distribution
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

            #do a tumble to see if this gets the tumble to occur
            else:
                self.status = 0
                self.timer = 0
                self.duration = self.getDuration(self.mean_tumble)

        #bacteria are hitting the right wall
        elif x > model_width:

            #adjust position
            x = model_width - epsilon

            #determine whether an arch or tangent collision occurs
            p = random.uniform(0, 1)

            #perform tangent colision otherwise arch
            if p >= arch_collision:
                self.ang = 180-self.ang

            else:
                self.status = 0
                self.timer = 0
                self.duration = self.getDuration(self.mean_tumble)

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

            else:
                self.status = 0
                self.timer = 0
                self.duration = self.getDuration(self.mean_tumble)

        #bacteria are hitting the bottom wall
        elif y > model_height:

            #adjust the position
            y = model_height-epsilon

            #determine whether a tangent or arch collision occurs
            p = random.uniform(0, 1)

            #perform tangent collision otherwise arch
            if p >= arch_collision:
                self.ang = 360 - self.ang

            else:
                self.status = 0
                self.timer = 0
                self.duration = self.getDuration(self.mean_tumble)

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
        Get the neighbours which are in the vicinity of the bacteria and see if they collide.
        Modelled as an inelastic collision. Bacteria pause until a new angle is selected.
        :return:
        """

        #get the neighbours of the bacteria
        vision = radius #vision is the radius of the bacteria
        colliders = self.model.space.get_neighbors(self.pos, vision, False)

        #if there are colliding bacteria generate new run angles
        if len(colliders) > 0:

            #generate a new run angle
            #self.ang = self.getTumbleAngle(self.ang_mean, self.ang_std) + self.ang
            #self.ang = self.ang % 360

            for collider in colliders:

                collider.velocity = 0
                #collider.ang = (self.getTumbleAngle(self.ang_mean, self.ang_std) +self.ang)
                #collider.ang = collider.ang %360

    def inelasticCollision(self, collider):
        """
        Reset the angle and velocity of two colliding cells
        TODO remove this function
        """

        #get the difference between the two angles
        ang_diff = (self.ang - collider.ang) % 360

        theta = 0 #placeholder for angle
        new_velocity = 0 #placeholder for velocity

        #apply the formula if the difference is greater than 90 degrees:
        if ang_diff > 90:

            #get angle
            phi = ang_diff - 90
            theta = np.arctan((collider.velocity * np.cos(np.deg2rad(phi)))/(collider.velocity*np.sin(np.deg2rad(phi))+self.velocity))
            theta = np.rad2deg(theta)

            #get velocity
            new_velocity = (collider.velocity*np.cos(np.deg2rad(phi)))/(2*np.sin(np.deg2rad(theta)))

        if ang_diff <= 90:

            #get angle
            phi = ang_diff
            theta = np.arctan((collider.velocity * np.sin(np.deg2rad(phi)))/(self.velocity + collider.velocity*np.cos(np.deg2rad(phi))))
            theta = np.rad2deg(theta)

            #get velocity
            new_velocity = (collider.velocity*np.sin(np.deg2rad(phi)))/(2*np.sin(np.deg2rad(theta)))

        #get the new angle and set
        new_angle = (180 + theta + self.ang) % 360
        self.ang = new_angle
        collider.ang = new_angle

        #set the new velocity to the agents
        self.velocity = new_velocity
        collider.velocity = new_velocity


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
                self.ang = self.getTumbleAngle(self.ang_mean, self.ang_std) + self.ang
                self.ang = self.ang %360
                # get the duration of the next run
                self.duration = self.getDuration(mean_run)
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
                        #self.duration = alpha * self.getDuration(mean_run)
                        self.status = 2

                else:

                    # if biasing already it is time to tumble
                    self.status = 0
                    # get the duration of the tumble
                    self.duration = self.getDuration(self.mean_tumble)

            # reset timer
            self.timer = 0

    def reverseStep(self):
        """
        Adjust timers for a run and reverse motility pattern.
        1 - running
        2 - extending the run
        :return:
        """

        if self.timer >= self.duration:

	    #reset the timer
            self.timer = 0 

            #if the run has already been extended then reverse
            if self.status == 2:
                self.reverseRun()
                self.status = 1 

            #check if the run should be extened
            elif self.status == 1 :

                # if not biasing movement already see if it can be extended
                #current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc

                #if it is increasing then continue
                if self.c_end > self.c_start:
                    self.duration = alpha * self.duration
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 2
			

                #if it is not increasing then reverse
                else:
                    self.reverseRun()
	
    def reverseFlickStep(self):
        """
        Adjust timer for a run and reverse motility pattern
         1 - flicking
         2 - reversing
         3 - extending the reverse
        """

        if self.timer >= self.duration:
            
            #reset the timer	
            self.timer = 0 
            
            if self.status == 1:
                #if currently flicking then change to a reverse
                self.reverseRun()
                self.status = 2

            elif self.status == 3:

                #if already extending it is time to flick
                self.flick()
                self.status = 1

            elif self.status == 2:
                #if currently reversing then see if reverse should be extended
                #current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc
		

                # if it is increasing then extend the run
                if self.c_end > self.c_start:
                    self.duration = alpha * self.duration 
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 3

                #if not then flick
                else:
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
        
        #get a new angle
        flick_ang = np.random.normal(90, self.flick_std,1) 
        flick_direction = random.randint(0,1)
        self.ang = (self.ang + flick_ang * flick_direction) % 360 
        
        #get a duration for the flick 
        self.duration = self.getDuration(mean_run)
        
        #get the velocity of the new run 
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)

    def step(self):
        """
        Perform operations to move bacteria for one tick
        """

        #if bacteria have just doubled reset the doubling timer
        if self.next_double < self.dt:
            self.next_double = np.random.normal(doubling_mean, doubling_std, 1)

        #check if the bacteria is about to collide
        #self.neighbourCollide()

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
        if self.pattern == 'reverse':
            self.reverseStep()
        if self.pattern == 'flick':
            self.reverseFlickStep()

	#uncomment only to make the figure of the run and tumble motlity pattern 
	#save the positions to a file

        if self.unique_id == 1:
            self.step_counter = self.step_counter + 1
            pos_list.append(self.pos)

            #save only every 10 second
            if self.step_counter % 1000 == 0:
                pos_df = pd.DataFrame({'position': pos_list})
                pos_df .to_csv('example_position_list_tumble.csv', index = False)
        






