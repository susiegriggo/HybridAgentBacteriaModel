"""
Tube model which inserts a number of bacteria set in server.py
The concentration of attractant is modelled using a partial differential equation.
The density of bacteria is approximated using kernel density estimation.
Multiple model objects can be created - allowing for multiple simulations to be run as replicates
"""

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from bacteria import Bacteria

import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.neighbors import KernelDensity
import time
from fastdist import fastdist
from collections import Counter

#set global variables for the tube model
D_c = 1.e-10 #diffusion coefficient of the nutrients
c_0= 6E-3 #intial concentration of molecules in the tube
beta = 5E-18 #number of moles of glucose consumed per bacteria per second
radius = 1*10E-4 #radius of bacteria in centimetres

#scaling parameters
tau = 1 #time scale
L = 1 #lengh scalin

class Tube(Model):
    """
    Creates tube mode. Handles scheduling of the agents and calculates the densities of bacteria.
    """

    def __init__(
        self,
        population=100,
        width_=100,
        height_=100,
        name = " ",
        pattern = "tumble",
        beta_star = False,
        c_star = False,

    ):

        """
        Create a model object
        Args:
            population: number of agents starting in the model
            width: width of the modelling space in cm
            height: height of the modelling space in cm
            name: name given as a prefix for the model files
            pattern: movement pattern to use in the model
            beta_star = dimensionless consumption term
            c_star = dimensionless starting concentration
        """

        self.population = int(population)
        self.width = width_
        self.height = height_
        self.prefix =  str(name)+'_pop'+str(self.population)
        self.pattern = pattern
        self.dx = 0.001 #size of grid increments
        self.dt = 0.01 #length of the timesteps in the model
        self.nx = int(width_/self.dx) #number of increments in x direction
        self.ny = int(height_/self.dx) #number of increments in y direction
        self.ticks = 1 #count the number of ticks which have elapsed
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width_, height_, False)

        #calculate the values for dimensionless parameters
        self.p_inf = self.population/(self.width*self.height) #starting density of bacteria
        self.D_star = (D_c*tau)/(L*L)

        #if value for c_star is not parsed calculated set to 1
        if not c_star:
            self.c_star = 1
        else:
            self.c_star = c_star

        #if a value for beta_star is not given calculate manually
        if not beta_star:
            self.beta_star = (beta*self.p_inf*tau)/(c_0*self.width*self.height)
        else:
            self.beta_star = beta_star

        #update the prefix with these terms
        self.prefix = self.prefix + '_betastar'+str(self.beta_star)+'_cstar'+str(self.c_star)

        #create agents
        self.make_agents()
        self.running = True

        #generate grid to solve the concentration over
        self.u0 = self.c_star * np.ones((self.nx+1, self.ny+1)) #starting concentration of bacteria
        self.u = self.u0.copy() #current concentration of bacteria

        #store the location of the band at each timepoint
        self.band_location = []  # intialise list to store the location of the band at each dt

        global dx, width, height, nx, ny
        dx = self.dx
        nx = self.nx
        ny = self.ny
        width = self.width
        height = self.height


    def make_agents(self):
        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.population):

            #have the initial position start at the centre of the y axis
            #x = self.width/1000
            x = self.width/2 #uncomment to position bacteria in the centre of the modelling space
            y = self.height/2

            #x = self.width/2
            pos = np.array((x, y))

            bacteria = Bacteria(
                i,
                self,
                pos,
                self.width,
                self.height,
                False,
                self.prefix,
                self.pattern,
            )
            self.space.place_agent(bacteria, pos)
            self.schedule.add(bacteria)


    def bacteriaReproduce(self):
        """
        Models the bacteria dividing and adding an extra agent at
        the same location as the bacterium which is dividing
        """

        # get a list containing all of the agents
        all_agents = self.schedule.agents
        agent_growthstate = np.array([all_agents[i].next_double for i in range(len(all_agents))])

        #identify which of these are less than dt
        grew = np.where(agent_growthstate < self.dt)

        if len(grew[0]) > 0:

            for i in range(len(grew[1])):

                #get the position of the new bacteria
                pos = all_agents[i].pos

                self.population = self.population + 1

                bacteria = Bacteria(
                    self.population+1,
                    self,
                    pos,
                    self.width,
                    self.height,
                    True,
                    self.prefix,
                    self.pattern,
                    2.41E-3,
                )
                self.space.place_agent(bacteria, pos)
                self.schedule.add(bacteria)


    def densityKernel(self, agent_positions):
        """
        Alternative to density kernel function above
        """
        #get a list of the x and y values
        x_list = [position[0] for position in agent_positions]
        y_list = [position[1] for position in agent_positions]
        xy = np.vstack([x_list, y_list])

        #calcualte the bandwidth for the computation
        d = xy.shape[0] #number of dimensions
        n = xy.shape[1]

        #calculate the bandwidth
        bw = scottsRule(x_list, y_list)

        #calculate the density kernel
        xx, yy, zz, = kde2D(x_list,y_list,bw, nx+1,ny+1)

        return zz

    def stepConcentration(self):
        """
        Update the concentration grid of the model depending on the current location of agents.
        Uses finite difference equations of Ficks law from Franz et al.
        """
        dx2, dy2 = self.dx * self.dx, self.dx * self.dx

        #get the positions of all agents
        all_agents = self.schedule.agents
        agent_positions = [all_agents[i].pos for i in range(len(all_agents))]

        #get the gaussian density kernel of the bacteria
        bacterial_density = self.densityKernel(agent_positions)

        #solve the partial differential equation model using the fintie difference method
        self.u[1:-1, 1:-1] = self.u0[1:-1, 1:-1] + self.D_star * self.dt * ((self.u0[2:, 1:-1] - 2 * self.u0[1:-1, 1:-1] + self.u0[:-2, 1:-1]) / dx2 + (
                        self.u0[1:-1, 2:] - 2 * self.u0[1:-1, 1:-1] + self.u0[1:-1, :-2]) / dy2) - self.dt * self.beta_star *self.population*bacterial_density[1:-1, 1:-1]

        # set such that the concentration cannot be lowered below zero
        self.u[self.u < 0] = 0
        # incorporate the bacteria into this using vectorisation
        self.u0 = self.u.copy()

        dens_df = pd.DataFrame(bacterial_density)

        #save updated versions of the density and concentration periodically
        if self.ticks % 100 == 0: #save every 100 ticks (i.e every 10 seconds)
            concfield_name = str(self.prefix)+'_concentration_field_'+str(self.ticks) + "_ticks.csv"
            densfield_name = str(self.prefix)+'_density_field_' +str(self.ticks) + "_ticks.csv"

            u_df = pd.DataFrame(self.u)
            conc_file = str(self.prefix)+'_concentration_field.csv'
            u_df.to_csv(conc_file, index = False)
            u_df.to_csv(concfield_name, index = False)
            dens_df.to_csv(densfield_name, index = False)

        #update band location with the current location of the chemotaxis band
        self.detectBand(dens_df)
        #save the band density
        #if self.ticks % 100 == 0: #save every 100 ticks (i.e every 1 second)
        #   band_name = str(self.name) + '_band_location_'+str(self.ticks)+"_ticks.csv"
        #   band_df = pd.DataFrame({'time': [self.dt*i for i in range(0,self.ticks)], "distance (cm)": self.band_location})
        #   band_df.to_csv(band_name, index = False)

    def detectBand(self, dens_df):
        """
        Detect the band in the tube by evaluating the mean bacterial density across each row
        """
        dens_df = dens_df.T
        col_means = dens_df.values.mean(axis=0)

        #get the column with the maximum density
        max_dens_dx = np.where(col_means == np.amax(col_means))

        #get the location relative to the size of the tube
        max_dens_loc = max_dens_dx[0][0] * self.dx

        #update the list of band locations
        self.band_location.append(max_dens_loc)

    def neighbourCollide(self):
        """
        Check if neighbours  colliding using a grid. If neighbours collide perform an inelastic collision.
        If there is two cells with the same angle consider these the same cell
        """

        #get the positions of all agents
        agent_positions = [agent.pos for agent in self.schedule.agents]
        agent_positions = np.array(agent_positions)

        #round the list of positions using the radius
        r = 4 #round to 4 decimals - consistent with 1 micron radius
        agent_pos_rounded = [(round(position[0], r),round(position[1], r)) for position in agent_positions]

        #see how many collision points occur
        position_counter = Counter(agent_pos_rounded)
        counter_df = pd.DataFrame.from_dict(position_counter, orient='index').reset_index()
        colliders = counter_df[counter_df[0] > 1]
        collider_points = colliders['index'].values

        #loop through the colliding points
        for point in collider_points:

            #get the indices corresponding to the point
            idx = [i for i, d in enumerate(agent_pos_rounded) if d == point]

            #if there are more than two agents involved in the collision just consider the closest two
            if len(idx) > 2:

                #get the positions of the colliding bacteria
                position_list = [self.schedule.agents[i].pos for i in idx]
                position_list = np.array(position_list)

                #get the 2 closest points out of these points
                dist_mat = fastdist.matrix_pairwise_distance(position_list, fastdist.euclidean, "euclidean", return_matrix=True)

                #get the index of the minimum
                idx = np.argwhere(dist_mat == np.min(dist_mat))[0]

            # perform the collision by swapping the angles (simulate an incidence angle)
            angle_0 = self.schedule.agents[idx[0]].ang
            angle_1 = self.schedule.agents[idx[1]].ang

            self.schedule.agents[idx[0]].ang = angle_1
            self.schedule.agents[idx[1]].ang = angle_0

    def step(self):

        #perform a step
        print('STEPPING')
        start = time.time()
        self.schedule.step()
        end = time.time()
        print(end - start)

        # correct for neighbouring bacteria which have collided
        # ignore the first tick as everything will collide
        if self.ticks > 1:
            print('CONCENTRATION')
            start = time.time()
            self.stepConcentration()
            end = time.time()
            print(end - start)
            print('NEIGHBOURS')
            start = time.time()
            self.neighbourCollide()
            end = time.time()
            print(end - start)

        # let bacteria reproduce
        print('REPRODUCING')
        start = time.time()
        self.bacteriaReproduce()
        end = time.time()
        print(end - start)

        #update the number of ticks which have occured
        self.ticks = self.ticks + 1
        if self.ticks % 100 == 0:
            print('TIME ELAPSED: '+ str(self.ticks*self.dt)+ ' seconds', flush = True)

def kde2D(x, y , bandwidth, xbins, ybins, **kwargs):

    #create a grid of the agent locations
    xx, yy = np.mgrid[0:width:xbins*1j, 0:height:ybins*1j]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    #calculate the density kernel
    kde_skl = KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', rtol=0)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))

    return xx, yy, np.reshape(z, xx.shape)

def scottsRule(x, y):
    """Compute bandwidth using Scotts rule. Note that in two-dimensions Scotts rule is equivalent to silvermans rule"""
    
    n =  len(x)
    std = np.array((np.std(x),np.std(y)))
    return np.mean(std)* n**(-1/6)












