"""
Creates a bacteria object which can be aded to a tube model
"""
import numpy as np
import random
import pandas as pd
from mesa import Agent

#set model parameters
radius = 10E-4  #size of bacteria in radians
D_rot = 0.062 #rational diffusion coefficient radians^2/s
D_rot = D_rot *10E-9 #convert rotational diffusion coefficient to the correct units
epsilon = 10E-16 #adjusts edges of the modelling space
alpha =  2 #bias of the bacteria to the nutrients it is consuming
doubling_mean = 360 #mean doubling time
doubling_std = 20 #standard deviation of doubling time
velocity_mean = 2.41E-3 #mean velocity in cm/s
velocity_std = 6E-4 #standard deviation of the velocity

#wall effects
arch_collision = 1  #probability of an arch collision
tangent_collision = 1 - arch_collision #probability of a tangential deflection collision

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

        """
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.pos = np.array(pos)
        self.model_name = model_name
        self.pattern = pattern

        #parameters universal to all motility patterns
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)
        self.mean_run = 1

        if self.pattern == 'tumble':
            self.ang_mean = 68
            self.ang_std = 37
            self.mean_tumble = 0.1
            self.duration = self.getDuration(self.mean_tumble)  # stores the time of the duration

        if self.pattern == 'reverse':
            self.reverse_std = 20  # standard deviation of angle
            self.status = 1  # 1 is running, 2 is extending a run
            self.duration = self.getDuration(self.mean_run)  # duration of the first run

        if self.pattern == 'flick':
            self.status = 1  # 1 is running, 2 is extending a run, 3 is flicking
            self.flick_std = 10  # standard deviation of a flick
            self.reverse_std = 2  # standard deviation of a reverse
            self.duration = self.getDuration(self.mean_run)  # duration of the first run

        self.status = 0 #determines whether running or tumbling. 0 is tumbling, 1 is running, 2 is extending a run
        self.dt = 0.01 #time for each tick in the simulation
        self.timer = 0  #traces where up to on the current run/tumble
        self.ang = 0 #angle for running

        self.pos_list = []
        self.ticks = 1

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
        Change the values on based on how large the modelling area is
        """
        x = pos[0]
        y = pos[1]

        #bacteria are hitting the left wall
        if x < 0:
            x = 0

            p = random.uniform(0, 1)

            if p > arch_collision:

                self.ang = self.ang+90

        #bacteria are hitting the right wall
        elif x > model_width:
            x = model_width-epsilon

            p = random.uniform(0, 1)

            if p > arch_collision: #tumbling at the walls
                self.ang = self.ang+90

        #bacteria are hitting the bottom wall
        if y < 0:
            y = 0

            p = random.uniform(0, 1)

            if p > arch_collision:

                self.ang = self.ang+90

        #bacteria are hitting the top wall
        elif y > model_height:
            y = model_height-epsilon

            p = random.uniform(0, 1)

            if p >  arch_collision:

                self.ang = self.ang+90

        return [x,y]

    def getConcentration(self):
        """
        Get the concentration of attractant at a location for the currently saved concentration field
        :return:
        """
        conc_file = str(self.model_name) + '_concentration_field.csv'
        conc_field = pd.read_csv(conc_file, sep=',')

        #get the number of columns and rows in the concentration field
        conc_nx = len(conc_field.columns) - 1
        conc_ny = len(conc_field) - 1

        #get the increase in each direction per unit time
        conc_dx = model_width/conc_nx
        conc_dy = model_height/conc_ny

        #find the number of decimal places to round the position to
        round_dx = np.log10(conc_dx)*-1
        round_dy = np.log10(conc_dy)*-1

        #get the concentration corresponding to this rounded concetration
        field_pos = [int(round(self.pos[0],int(round_dx))/conc_dx), int(round(self.pos[1], int(round_dy))/conc_dy)]

        #get this position in the concentration dataframe
        current_pos = conc_field.loc[field_pos[1]][field_pos[0]]
        return current_pos

    def tumbleStep(self):

        # check whether the duration is up
        if self.timer >= self.duration:

            # if currently tumbling change to run
            if self.status == 0:
                self.status = 1
                # generate a new running angle
                self.ang = (self.getTumbleAngle(self.ang_mean, self.ang_std) + self.ang) % 360
                # get the duration of the next run
                self.duration = self.getDuration(self.mean_run)
                self.velocity = np.random.normal(velocity_mean, velocity_std, 1)

            # if its not tumbling its either extending or running
            else:

                if self.status != 2:
                    # if not biasing movement already see if it can be biased
                    current_conc = self.getConcentration()
                    self.c_start = self.c_end
                    self.c_end = current_conc

                    if self.c_end > self.c_start:
                        self.duration = alpha * self.getDuration(self.mean_run)
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

            # reset the timer
            self.timer = 0

            # if the run has already been extended then reverse
            if self.status == 2:
                self.reverseRun()
                self.status = 1

                # check if the run should be extened
            elif self.status == 1:

                # if not biasing movement already see if it can be extended
                # current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc

                # if it is increasing then continue
                if self.c_end > self.c_start:
                    self.duration = alpha * self.duration
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 2


                # if it is not increasing then reverse
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

            # reset the timer
            self.timer = 0

            if self.status == 1:
                # if currently flicking then change to a reverse
                self.reverseRun()
                self.status = 2

            elif self.status == 3:

                # if already extending it is time to flick
                self.flick()
                self.status = 1

            elif self.status == 2:
                # if currently reversing then see if reverse should be extended
                # current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc

                # if it is increasing then extend the run
                if self.c_end > self.c_start:
                    self.duration = alpha * self.duration
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 3

                # if not then flick
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

        # get a new angle
        flick_ang = np.random.normal(90, self.flick_std, 1)
        flick_direction = random.randint(0, 1)
        self.ang = (self.ang + flick_ang * flick_direction) % 360

        # get a duration for the flick
        self.duration = self.getDuration(mean_run)

        # get the velocity of the new run
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)

    def reverseStep(self):
        """
        Adjust timers for a run and reverse motility pattern.
        1 - running
        2 - extending the run
        :return:
        """

        if self.timer >= self.duration:

            # reset the timer
            self.timer = 0

            # if the run has already been extended then reverse
            if self.status == 2:
                self.reverseRun()
                self.status = 1

            # check if the run should be extened
            elif self.status == 1:

                # if not biasing movement already see if it can be extended
                # current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc

                # if it is increasing then continue
                if self.c_end > self.c_start:
                    self.duration = alpha * self.duration
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 2


                # if it is not increasing then reverse
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

            # reset the timer
            self.timer = 0

            if self.status == 1:
                # if currently flicking then change to a reverse
                self.reverseRun()
                self.status = 2

            elif self.status == 3:

                # if already extending it is time to flick
                self.flick()
                self.status = 1

            elif self.status == 2:
                # if currently reversing then see if reverse should be extended
                # current_conc = self.pos[0]
                current_conc = self.getConcentration()
                self.c_start = self.c_end
                self.c_end = current_conc

                # if it is increasing then extend the run
                if self.c_end > self.c_start:
                    self.duration = alpha * self.duration
                    # self.duration = alpha * self.getDuration(mean_run)
                    self.status = 3

                # if not then flick
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

        # get a new angle
        flick_ang = np.random.normal(90, self.flick_std, 1)
        flick_direction = random.randint(0, 1)
        self.ang = (self.ang + flick_ang * flick_direction) % 360

        # get a duration for the flick
        self.duration = self.getDuration(self.mean_run)

        # get the velocity of the new run
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)

    def step(self):
        """
        Move the bacteria accordingly
        """

        #if bacteria have just doubled reset the doubling timer
        if self.next_double < self.dt:
            self.next_double = np.random.normal(doubling_mean, doubling_std, 1)

        #get the current timestep in the run/tumble
        step_num = int(self.timer/self.dt)
        x_new = self.pos[0] + self.velocity*self.status*np.cos(np.deg2rad(self.ang))*self.dt#+np.sqrt(2*D_rot*self.dt)*self.W_x
        y_new = self.pos[1] + self.velocity*self.status*np.sin(np.deg2rad(self.ang))*self.dt#+np.sqrt(2*D_rot*self.dt)*self.W_y
        new_pos = [x_new[0], y_new[0]]

        new_pos = self.checkCollision(new_pos)
        self.pos = new_pos
        self.model.space.move_agent(self, self.pos)

        #add this timestep to the timer
        self.timer = self.timer + self.dt
        #update the time until the next double
        self.next_double = self.next_double - self.dt

        #update the wiener processes for rotational diffusion
        self.W_x = self.W_x + np.sqrt(self.dt) * np.random.normal(0,1,1)
        self.W_y = self.W_y + np.sqrt(self.dt) * np.random.normal(0,1,1)

        # check if the status of the cell needs to be changed
        if self.pattern == 'tumble':
            self.tumbleStep()

        #add the tick just completed
        self.ticks = self.ticks + 1











