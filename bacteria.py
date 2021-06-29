import numpy as np
import random
import pandas as pd

from mesa import Agent

# calculate the rotational diffusion coefficient
# calculated as a sphere representing E. coli
#K = 1.38E-23  # Boltzman's coefficient
#T = 298  # temperature in kelvin
#vis = 0.027  # g/cm viscosity
radius = 10E-4  # cm
#f_sphere = 8 * np.pi * vis * radius  # frictional drag coefficient for a sphere
#D_rot = (K * T) / f_sphere
D_rot = 0.062 #rational diffusion coefficient radians^2/s
epsilon = 10E-10 #adjusts edges of the modelling space

alpha =  1 #bias of the bacteria to the nutrients it is consuming
doubling_mean = 27000
doubling_std = 120
doubling_mean = 360
doubling_std = 20
velocity_mean = 2.41E-3 #mean velocity in cm/s
#velocity_std = 6.8E-8
velocity_std = 0

#wall effects
arch_collision = 0.4 #probability of an arch collision
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
        doubling,
    ):
        """
        Create a new Bacteria

        Args:
            unique_id: Unique agent identifier.
            pos: Starting position

        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.ang_mean = 68
        self.ang_std = 37
        self.velocity = np.random.normal(velocity_mean, velocity_std, 1)
        self.mean_run = 1
        self.mean_tumble = 0.1
        self.status = 0 #determines whether running or tumbling. 0 is tumbling, 1 is running
        self.duration = self.getDuration(self.mean_tumble) #stores the time of the duration
        self.dt = 0.01 #time for each tick in the simulation
        self.timer = 0  #traces where up to on the current run/tumble
        self.ang = 0 # angle for running
        self.x_wiener = self.wienerProcess(20, self.dt)
        self.y_wiener = self.wienerProcess(20, self.dt)
        self.c_start = 0 #concentration of attractant at the start of the run
        self.c_end = 0 #concentration of attractant at the end of the run

        if doubling == False:
            self.next_double = np.random.uniform(0,doubling_mean, 1) #select the reproduction rate from a uniform distribution
        else:
            self.next_double = np.random.normal(doubling_mean, doubling_std, 1)

        self.F = [0,0] #variable which when not zero represents bacteria colliding with wall

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

    def wienerProcess(self, T, dt):
        """
        Used to simulate Brownian motion
        dt - tick time
        T - number of seconds to generate the Wiener process
        :return:
        """

        #number of discrete timesteps
        N = int(T/dt)

        random_increments = np.random.normal(0, 1, N)*np.sqrt(dt) #the epsilon values
        wiener_process = np.cumsum(random_increments) #calculate the wiener process
        wiener_process = np.insert(wiener_process, 0, 0) #insert the intial condition

        return wiener_process

    def checkCollision(self, pos):
        """
        Check if the bacteria collide with the side of the tube.
        If bacteria collide shorten the run and induce a tumble.
        Change the values on based on how large the modelling area is
        """
        x = pos[0]
        y = pos[1]

        if x < 0 or x > model_width or y < 0 or y > model_height:

        # draw from a uniform distribution to determine whether tangental collision occurs
            collision_type = random.uniform(0, 1)
            if collision_type <= tangent_collision:
                #end the run prematurely and change to a tangent
                #TODO could change this to act like a bounce rather than introducing a tumble early
                self.timer = self.duration

        #bacteria are hitting the left wall
        if x < 0:
            x = 0

        #bacteria are hitting the right wall
        elif x > model_width:
            x = model_width-epsilon

        #bacteria are hitting the bottom wall
        if y < 0:
            y = 0

        #bacteria are hitting the top wall
        elif y > model_height:
            y = model_height-epsilon

        return [x,y]

    def getConcentration(self):
        """
        Get the concentration of attractant at a location for the currently saved concentration field
        :return:
        """
        conc_field = pd.read_csv('concentration_field.csv', sep = ',')
        #get the number of columns and rows in the concentration field
        conc_nx = len(conc_field.columns) - 1
        conc_ny = len(conc_field) - 1

        #get the increase in each direction per unit time
        conc_dx = model_width/conc_nx
        conc_dy = model_height/conc_ny

        #find the number of decimal places to round the position to
        round_dx = np.log10(conc_dx)*-1
        round_dy = np.log10(conc_dy)*-1

        #round the agents position
        #rounded_pos = [round(self.pos[0],int(round_dx)), round(self.pos[1], int(round_dy))]

        #get the concentration corresponding to this rounded concetration
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

        if len(colliders) > 0:

            #generate a new run angle
            self.ang = self.getAngle(self.ang_mean, self.ang_std)

            #give the colliders new angles as well
            for collider in colliders:
                collider.ang = self.getAngle(self.ang_mean, self.ang_std)

    def step(self):
        """
        Move the bacteria accordingly
        """

        #check if the bacteria is about to collide
        self.neighbourCollide()
        #get the current timestep in the run/tumble
        step_num = int(self.timer/self.dt)
        x_new = self.pos[0] + self.velocity*self.status*np.cos(np.deg2rad(self.ang))*self.dt+np.sqrt(2*D_rot*self.dt)*self.x_wiener[step_num] #- self.velocity*(self.F[0])
        y_new = self.pos[1] + self.velocity*self.status*np.sin(np.deg2rad(self.ang))*self.dt+np.sqrt(2*D_rot*self.dt)*self.y_wiener[step_num]#- self.velocity*(self.F[1])
        new_pos = [x_new[0], y_new[0]]

        new_pos = self.checkCollision(new_pos)
        self.pos = new_pos #added
        self.model.space.move_agent(self, self.pos)

        #add this timestep to the timer
        self.timer = self.timer + self.dt
        #update the time until the next double
        self.next_double = self.next_double - self.dt

        #if bacteria have just doubled reset the doubling timer 
        if self.next_double < self.dt:
            self.next_double = np.random.normal(doubling_mean, doubling_std, 1)
        
	#check whether the duration is up
        if self.timer >= self.duration:

            # reset collision variable
            self.F = [0, 0]

            #if currently tumbling change to run
            if self.status == 0:
                self.status = 1
                #generate a new running angle
                self.ang = self.getAngle(self.ang_mean, self.ang_std)
                #get the duration of the next run
                self.duration = self.getDuration(self.mean_run)
                #get the wiener processes for the next run
                #generate for 10 seconds in case the run is extended
                self.x_wiener = self.wienerProcess(20, self.dt)
                self.y_wiener = self.wienerProcess(20, self.dt)

            #if currently running see if running in the direction of nutrients
            elif self.status == 1:

                #get the concentration of attractant and update start and end
                current_conc = self.getConcentration()
                #current_conc = self.pos[0] #debugging just use x coordinate
                self.c_start = self.c_end
                self.c_end = current_conc

                #if moving in the direction of nutrients continue to run
                if self.c_end > self.c_start:

                    #if not located along a wall
                    if new_pos[0]!= 0 and new_pos[0] != model_width-epsilon and new_pos[1] != 0 and new_pos[1] != model_height-epsilon:
                        #generate a duration for the next run
                        self.duration = alpha*self.getDuration(self.mean_run)
                        #don't update the angle or wiener proccess as continuing in same direction

                else:
                    #if moving in a 'bad' direction tumble
                    self.status = 0
                    #get the duration of the tumble
                    self.duration = self.getDuration(self.mean_tumble)

            #reset timer
            self.timer = 0










