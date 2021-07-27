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
        dt = 0.01, 
        dx_ = 0.1,
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
        self.dx = dx_  #size of grid increments
        self.dt = dt  #length of the timesteps in the model
        self.nx = int(width_/self.dx) #number of increments in x direction
        self.ny = int(height_/self.dx) #number of increments in y direction
        self.ticks = 1 #count the number of ticks which have elapsed
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width_, height_, False)
        
        #set the innoculation point
        self.innoculate = innoculationPoint(self.width, self.height)

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

        #generate a list to store the chemotactic motility coefficient 
        self.cmc_list = [] 
 
        #create agents
        self.make_agents()
        self.running = True

        #generate grid to solve the concentration over
        self.u0 = self.c_star * np.ones((self.nx+1, self.ny+1)) #starting concentration of bacteria
        self.u = self.u0.copy() #current concentration of bacteria

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
 
        #loop to add bateria at to the space 
        for i in range(self.population):
            
            #create a baterial agent 
            bacteria = Bacteria(
                i,
                self,
                self.innoculate,
                self.width,
                self.height,
                False,
                self.prefix,
                self.pattern,
            )

            #add this bacterial agent to the modelling space 
            self.space.place_agent(bacteria, self.innoculate)
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

        #loop through the bacteria which are reproducing
        if len(grew[0]) > 0:

            for i in range(len(grew[1])):

                #get the position of the new bacteria
                pos = all_agents[i].pos
                
                #adjust the population counter 
                self.population = self.population + 1
                
                #create a new bacteria agent 
                bacteria = Bacteria(
                    self.population+1,
                    self,
                    pos,
                    self.width,
                    self.height,
                    True,
                    self.prefix,
                    self.pattern,
                )
                
                #add this agent to the modelling space 
                self.space.place_agent(bacteria, pos)
                self.schedule.add(bacteria)


    def densityKernel(self, agent_positions):
        """
        Use a multivariate density kernel to approximate the concentration of bacteria at each grid point 
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

        #calculate components needed for the discretisation 
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

        #replcae the stored conccentration field with the newly generate concnetration field 
        self.u0 = self.u.copy()

        #save updated versions of the density and concentration periodically
        if self.ticks % 100 == 0: #save every 100 ticks (i.e every 10 seconds)
            concfield_name = str(self.prefix)+'_concentration_field_'+str(self.pattern)+'_pattern_'+'_dt'+str(self.dt)+'_'+str(self.ticks) + "_ticks.csv"
            densfield_name = str(self.prefix)+'_density_field_' +str(self.pattern)+'_pattern_'+'_dt'+str(self.dt)+'_'+str(self.ticks) + "_ticks.csv"

            #save the concentration field to a file 
            u_df = pd.DataFrame(self.u)
            u_df.to_csv(concfield_name, index = False)
            
            #save the density field to a file 
            dens_df = pd.DataFrame(bacterial_density)
            dens_df.to_csv(densfield_name, index = False)

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

        #get the points where each of these collisions occur 
        colliders = counter_df[counter_df[0] > 1]
        collider_points = colliders['index'].values    
       
        #loop through the colliding points
        for point in collider_points:
            
            #get the indices corresponding to the point
            idx = [i for i, d in enumerate(agent_pos_rounded) if d == point]
            
            #if the collision is only between a few cells determine which two cells will collide first 
            if len(idx)<6:
    
                    #get the indices corresponding to the point
                    idx = [i for i, d in enumerate(agent_pos_rounded) if d == point]

                    #if there are more than two agents involved in the collision just consider the closest two
                    if 2<len(idx):
                         
                        #get the positions of the colliding bacteria
                        position_list = [self.schedule.agents[i].pos for i in idx]
                        position_list = np.array(position_list)

                        #get the 2 closest points out of these points
                        dist_mat = fastdist.matrix_pairwise_distance(position_list, fastdist.euclidean, "euclidean", return_matrix=True)

                        #get the index of the minimum
                        idx = np.argwhere(dist_mat == np.min(dist_mat))[0]

                    #perform the collision by swapping the angles (simulate an incidence angle)
                    angle_0 = self.schedule.agents[idx[0]].ang
                    angle_1 = self.schedule.agents[idx[1]].ang

                    self.schedule.agents[idx[0]].ang = angle_1
                    self.schedule.agents[idx[1]].ang = angle_0

            #if the collision contains more cells then generate random angles  
            else:
                
                angles = np.random.uniform(0,360,len(self.schedule.agents))
                map(mapAngle, self.schedule.agents, angles)  

            # perform the collision by swapping the angles (simulate an incidence angle)
            angle_0 = self.schedule.agents[agent_list[0]].ang
            angle_1 = self.schedule.agents[agent_list[1]].ang

            self.schedule.agents[agent_list[0]].ang = angle_1
            self.schedule.agents[agent_list[1]].ang = angle_0

    def cmcUpdate(self): 
        """
        Update cmc_list the current chemotactic motility coefficient
        """ 

        #get the current cmc 
        this_cmc = self.cmc() 

        #update the cmc_list 
        self.cmc_list = self.cmc_list.appent(this_cmc)

        #every 100 ticks save the cmc data to a dataframe 
        if self.ticks % 100 == 0: 
            
            #generate the corresponding time labels in seconds 
            df_labels = [i*self.dt for i in range(self.ticks+1)]

            #assemble into a dataframe 
            cmc_df = pd.DataFrame({'time (seconds)':df_labels, 'CMC': self.cmc_list})

            #get the prefix to save the cmc_df 
            cmc_prefix = str(self.prefix)+'cmc_df_'+str(self.pattern)+'_pattern_'+'_dt'+str(self..dt)+'_'+str(self.ticks)+'_ticks.csv'
            
            #save the cmc_df 
            cmc_df.to_csv(cmc_prefix, index = False )

    def cmc(self):
        """
        Calculate the Chemotactic Motility Coefficient for the current agent positions
        Just consider the x position (1-d space)
        """

        #get the list of agent positions
        agent_positions = [agent.pos[0] for agent in self.schedule.agents]
        agent_positions = np.array(agent_positions)

        #determine the current mean location in the x direction
        mean_pos = np.mean(agent_positions)

        #get the x coordinate of the innoculation point
        inn_x = self.innoculate[0]

        #calculate the cmc
        cmc = (mean_pos - inn_x)/(0.5*self.width)

        return cmc


    def probDensUpdate(self): 
        """
        Generate a dataframe of the probability density and save to a csv file 
        """

        #get the probability distribution 
        this_probdens = self.probDens()

        #get the prefix to save the density dataframe 
        probdens_prefix = str(self.prefix)+'probdens_df_'+str(self.pattern)+'_pattern_'+'_dt'+str(self..dt)+'_'+str(self.ticks)+'_ticks.csv'

        #save the dataframe 
        this_probdens.to_csv(probdens_prefix, index = False)

    def probDens(self):
        """
        Calculate the normalised density of bacteria for the current time point.
        Alternative to a heatmap but only considered in one-dimension. 
        Could be normalised for figures 
        """

        #get the x coordinate of all agents
        agent_positions = [agent.pos[0] for agent in self.schedule.agents]
        agent_positions = np.array(agent_positions) 
         
        #generate the count of bacteria in each grid point along the x axis 
        hist_values = np.histogram(agent_positions, bins = self.nx+1)[0] # TOOD check this 
        
        #change these values to the density 
        density_values = hist_values/(self.height*self.dx)

        #get the labels corresponding to these density values 
        density_labels = [n*self.dx for n in range(self.nx+1)]
        density_labels = np.array(density_labels)
       
        #create a dataframe 
        dist_df = pd.DataFrame({'x position': density_labels, 'density':density_values})
        
        return dist_df      
    
    def step(self):
        """
        Combine methods to do one step of the model
        """

        #perform a step 
        print('STEP') 
        start = time.time() 
        self.schedule.step()
        end = time.time() 
        print(end-start)

        #ignore the first tick 
        if self.ticks > 1:
            print('CONCENTRATION') 
            start = time.time() 
            #evalute the concentration field    
            self.stepConcentration()
            end = time.time()
            print(end-start)


            #correct for colliding cells 
            print('NEIGHBOURS') 
            start = time.time() 
            #self.neighbourCollide()
            end = time.time()
            print(end-start)


        #update bacterial reproduction
        print('REPRODUCE') 
        start = time.time()  
        self.bacteriaReproduce()
        end = time.time()
        print(end-start)

        #add the cmc to a list 
        self.cmcUpdate()

        #if the model is a tube save the density every 100 ticks 
        if self.width > self.height: 
            if self.ticks % 100 == 0:  
                self.probDensUpdate() 

        #update the number of ticks which have occured
        self.ticks = self.ticks + 1
        if self.ticks % 100 == 0:
            print('TIME ELAPSED: '+ str(self.ticks*self.dt)+ ' seconds', flush = True)

def innoculationPoint(width, height): 
    """
    Determine the innoulation point of the simulation based on the width and height parsed
    """ 
    
    #if the space is square innoculate in the centre
    x = 0
    if width == height:
        x = width/2

    #if rectangular modelling space innoculate the left edge 
    else:
        x = width/1000

    #innoculate in the centre of the y direction
    y = height/2
    
    return np.array((x,y))

def kde2D(x, y , bandwidth, xbins, ybins, **kwargs):
    """ 
    Calculate the Gausian density kernel using the current location of the bacterial cells 
    """ 

    #create a grid of the agent locations
    xx, yy = np.mgrid[0:width:xbins*1j, 0:height:ybins*1j]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

    #add the posiitions of bacteria to the grid 
    xy_train = np.vstack([y, x]).T

    #calculate the density kernel
    kde_skl = KernelDensity(bandwidth=bandwidth, algorithm='kd_tree', rtol=0.01)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))

    return xx, yy, np.reshape(z, xx.shape)

def scottsRule(x, y):
    """
    Compute bandwidth using Scotts rule. Note that in two-dimensions Scotts rule is equivalent to silvermans rule
    """
    
    #get the number of points to be considered for the kernel 
    n =  len(x)

    #calculate the standard deviation 
    std = np.array((np.std(x),np.std(y)))
    
    return np.mean(std)*n**(-1/6)

def mapAngle(agent, angle):
    """
    Helper function to quickly generate new angles for a set of agents
    """

    agent.ang = (self.ang + angle) % 360
    

