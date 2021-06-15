"""
Bacteria run and tumble
=============================================================
A Mesa implementation modified from Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.

There was an issue with the agents position position matrix occasionally not being positive definite
The gaussian_kde class has been modified in scipy such the covariance matrix is modified suh that epsilon * np.eye(D).
This has the effects of adding epsilon to each one of the eigenvalues. As episilon is small this doesn't introduce much error.

Chemical field PDE is solved using implicit discretionization. Can be demonstrated that it is stable
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
import sys

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from bacteria import Bacteria

#set global variables for the tube model
D_c = 1.e-10 #diffusion coefficient of the nutrients
c_0= 5.56E-3 #intial concentration of glucose in the tube
beta = 5E-18 #number of moles of glucose consumed per bacteria per second
radius = 10E-4  #radius of bacteria in centimetres
gamma = 0.3 #how far the bacteria get nutrients from the outside of the radius
tau = 1 #time scale
L = 1 #lengh scaling
Na = 1000
p_inf =Na/(15*1) #todo - starting cells/cm^2

#calculate the values for non-dimensionalised parameters
D_star = (D_c*tau)/(L*L)
c_star = 1
beta_star = (beta*p_inf*tau)/c_0

doubling_mean = 27000 #mean doubling time of bacteria in seconds in 0.1% glucose
doubling_std = 120 #std of doubling time of bacteria in seconds in 0.1% glucose

class Tube(Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        population=100,
        width=100,
        height=100,

    ):
        """

        Args:
            population: Number of Bacteria
            width, height: Size of the space."""
        self.population = population
        self.width = width
        self.height = height
        self.dx = 0.01 #set the size of the increments in the x and y directions
        self.dy = 0.01
        self.dt = 0.1 #length of the timesteps in the model
        self.nx = int(width/self.dx)
        self.ny = int(height/self.dy)
        #add concentration field here
        self.u0 = c_star * np.ones((self.nx+1, self.ny+1))
        self.u = self.u0.copy()
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.make_agents()
        self.running = True

    def make_agents(self):

        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.population):

            #have the initial position start at the centre of the y axis
            x = 0
            y = self.height/2


            pos = np.array((x, y))
            bacteria = Bacteria(
                i,
                self,
                pos,
                self.width,
                self.height,
                False,
            )
            self.space.place_agent(bacteria, pos)
            self.schedule.add(bacteria)



    def bacteriaReproduce(self):
        """
        Models the bacteria dividing and adding an extra agent at
        the same location as the bacterium which is dividing
        :return:
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
                )
                self.space.place_agent(bacteria, pos)
                self.schedule.add(bacteria)

                #reset the growth status of the bacteria which just doubled
                #all_agents[i].next_double = np.random.normal(doubling_mean, doubling_std, 1)


    def densityKernel(self):
        """
        Use the Gaussian density kernel to calculate the density of bacteria in the tube

        Uses a multivariate kernel estimate. Estimates the bandwidth
        :return:
        """

        # get a list containing all of the agents
        all_agents = self.schedule.agents

        # get the postions of these agents
        agent_positions = [all_agents[i].pos for i in range(len(all_agents))]
        #agent_positions = np.array(agent_positions)

        #determine how many decimals to round to
        r= round(-1 * np.log10(self.dx))

        #create a grid to calculate the density with
        #dens_grid = [[[np.round(i * self.dx, r), np.round(j * self.dy, r)] for i in range(self.nx + 1)] for j in range(self.ny + 1)]

        #create a dataframe to store the bacteria density
        #kernel_sum = np.ones((self.ny+1, self.nx+1))

        #width of the kernel
        #std = 5

        #calculate the sum of kernels
        #for bacteria in agent_positions:
        #    bacteria_kernel = [
        #        [kernel(np.linalg.norm(np.array(dens_grid[j][i]) - np.array(bacteria)), std) for i in range(self.nx+1)] for j
        #        in range(self.ny+1)]
        #    kernel_sum = kernel_sum + bacteria_kernel

        #return kernel_sum

        #generate a grid
        X, Y = np.mgrid[0:self.width:(self.nx + 1) * 1j, 0:self.height:(self.ny + 1) * 1j]
        #get the coordinates in the grid
        positions = np.vstack([X.ravel(), Y.ravel()])

        dens = sm.nonparametric.KDEMultivariate(data = agent_positions, var_type = 'cc')

        #get the bandwidth matrix
        bw = dens.bw

        #get the density at each grid point
        bact_dens = dens.pdf(positions)
        bact_dens = np.reshape(bact_dens.T, X.shape)

        if np.isnan(np.min(bact_dens)) == True:
            bact_dens = np.zeros((bact_dens.shape))

        return bact_dens.T

    def stepConcentration(self):
        """
        Update the concentration grid of the model depending on the current location of agents.
        Uses finite difference equations of Ficks law from Franz et al.
        :return:
        """
        dx2, dy2 = self.dx * self.dx, self.dy * self.dy

        #get the gaussian density kernel of the bacteria
        bacterial_density = self.densityKernel().T

        #update the concentration field
        #self.u[1:-1, 1:-1] = self.u0[1:-1, 1:-1] + D_c * self.dt * ((self.u0[2:, 1:-1] - 2 * self.u0[1:-1, 1:-1] + self.u0[:-2, 1:-1]) / dx2 + (
        #                self.u0[1:-1, 2:] - 2 * self.u0[1:-1, 1:-1] + self.u0[1:-1, :-2]) / dy2)
        #self.u[1:-1, 1:-1] = self.u[1:-1, 1:-1] - self.dt * beta * bacterial_density[1:-1, 1:-1]

        a = self.population/Na #scaling the density by the number of agents
        self.u[1:-1, 1:-1] = self.u0[1:-1, 1:-1] + D_star * self.dt * ((self.u0[2:, 1:-1] - 2 * self.u0[1:-1, 1:-1] + self.u0[:-2, 1:-1]) / dx2 + (
                        self.u0[1:-1, 2:] - 2 * self.u0[1:-1, 1:-1] + self.u0[1:-1, :-2]) / dy2) - self.dt * beta_star * self.u0[1:-1, 1:-1]*bacterial_density[1:-1, 1:-1]

        # set such that the concentration cannot be lowered below zero
        self.u[self.u < 0] = 0
        # incorporate the bacteria into this using vectorisation
        self.u0 = self.u.copy()

        #have the concentration save to a file such that it can be read by the agents
        u_df = pd.DataFrame(self.u)
        u_df.to_csv('concentration_field.csv', index = False)

        dens_df = pd.DataFrame(bacterial_density)
        dens_df.to_csv('density_field.csv', index = False )

    def step(self):
        self.schedule.step()
        self.stepConcentration()
        self.bacteriaReproduce()

def kernel(value, std):
    """
    Calculate the Gaussian kernel given a dataframe and standard deviation
    :param value:
    :param std:
    :return:
    """
    K = (1/(np.sqrt(2*np.pi*std*std)))*np.exp((-value*value)/(2*std*std))
    return K





