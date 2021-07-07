"""
Tube model which inserts a number of bacteria set in server.py
The concentration of attractant is modelled using a partial differential equation.
The density of bacteria is approximated using kernel density estimation.

Multiple model objects can be created - allowing for multiple simulations to be run as replicates
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from bacteria import Bacteria

#set global variables for the tube model
D_c = 1.e-10 #diffusion coefficient of the nutrients
c_0= 5.56E-3 #intial concentration of glucose in the tube
beta = 5E-18 #number of moles of glucose consumed per bacteria per second
radius = 10E-4	#radius of bacteria in centimetres

#scaling parameters
tau = 1 #time scale
L = 1 #lengh scaling
p_inf = 1 #population scaling


class Tube(Model):
	"""
	Creates tube mode. Handles scheduling of the agents and calculates the densities of bacteria.
	"""

	def __init__(
		self,
		population=100,
		width=100,
		height=100,
		name = " "

	):
		self.population = population
		self.width = width
		self.height = height
		self.name = name
		self.dx = 0.001 #size of grid increments
		self.dt = 0.01 #length of the timesteps in the model
		self.nx = int(width/self.dx) #number of increments in x direction
		self.ny = int(height/self.dx) #number of increments in y direction
		self.ticks = 1 #count the number of ticks which have elapsed
		self.schedule = RandomActivation(self)
		self.space = ContinuousSpace(width, height, False)
		self.make_agents()
		self.running = True 

		#calculate the values for dimensionless parameters
		self.D_star = (D_c*tau)/(L*L)
		self.c_star = 1
		self.beta_star = (beta*p_inf*tau)/(c_0*self.width*self.height)
		self.beta_star = 1E-8 #practise placeholder parameter

		#generate grid to solve the concentration over 
		self.u0 = self.c_star * np.ones((self.nx+1, self.ny+1)) #starting concentration of bacteria
		self.u = self.u0.copy() #current concentration of bacteria - updating through each timestep


		#store the location of the band at each timepoint
		self.band_location = []  # intialise list to store the location of the band at each dt

	def make_agents(self):

		"""
		Create self.population agents, with random positions and starting headings.
		"""
		for i in range(self.population):

			#have the initial position start at the centre of the y axis
			x = 0
			y = self.height/2


			pos = np.array((0.00035, y))

			bacteria = Bacteria(
				i,
				self,
				pos,
				self.width,
				self.height,
				False,
				self.name,
				"tumble"
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
					self.name,
					"tumble"
				)
				self.space.place_agent(bacteria, pos)
				self.schedule.add(bacteria)

	def densityKernel(self):
		"""
		Use the Multivariate Gaussian density kernel to calculate the density of bacteria in the tube

		Uses a multivariate kernel estimate. Estimates the bandwidth using Scotts rule.
		"""

		# get a list containing all of the agents
		all_agents = self.schedule.agents

		# get the postions of these agents
		agent_positions = [all_agents[i].pos for i in range(len(all_agents))]

		#determine how many decimals to round to
		r= round(-1 * np.log10(self.dx))

		#generate a grid
		X, Y = np.mgrid[0:self.width:(self.nx + 1) * 1j, 0:self.height:(self.ny + 1) * 1j]
		#get the coordinates in the grid
		positions = np.vstack([X.ravel(), Y.ravel()])

		dens = sm.nonparametric.KDEMultivariate(data = agent_positions, var_type = 'cc')

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
		"""
		dx2, dy2 = self.dx * self.dx, self.dx * self.dx

		#get the gaussian density kernel of the bacteria
		bacterial_density = self.densityKernel().T

		self.u[1:-1, 1:-1] = self.u0[1:-1, 1:-1] + self.D_star * self.dt * ((self.u0[2:, 1:-1] - 2 * self.u0[1:-1, 1:-1] + self.u0[:-2, 1:-1]) / dx2 + (
						self.u0[1:-1, 2:] - 2 * self.u0[1:-1, 1:-1] + self.u0[1:-1, :-2]) / dy2) - self.dt * self.beta_star *self.population*bacterial_density[1:-1, 1:-1]

		# set such that the concentration cannot be lowered below zero
		self.u[self.u < 0] = 0
		# incorporate the bacteria into this using vectorisation
		self.u0 = self.u.copy()

		#have the concentration save to a file such that it can be read by the agents
		u_df = pd.DataFrame(self.u)
		conc_file = str(self.name)+'_concentration_field.csv'
		u_df.to_csv(conc_file, index = False)

		dens_df = pd.DataFrame(bacterial_density)

		#save updated versions of the density and concentration periodically
		if self.ticks % 10 == 0: #save every 100 ticks (i.e every 10 seconds)
			concfield_name = str(self.name)+'_concentration_field_'+str(self.ticks) + "_ticks.csv"
			densfield_name = str(self.name)+'_density_field_' +str(self.ticks) + "_ticks.csv"
			u_df.to_csv(concfield_name, index = False)
			dens_df.to_csv(densfield_name, index = False)

		#update band location with the current location of the chemotaxis band
		self.detectBand(dens_df)
		#save the band density
		#if self.ticks % 100 == 0: #save every 100 ticks (i.e every 1 second)
		#	band_name = str(self.name) + '_band_location_'+str(self.ticks)+"_ticks.csv"
		#	band_df = pd.DataFrame({'time': [self.dt*i for i in range(0,self.ticks)], "distance (cm)": self.band_location})
		#	band_df.to_csv(band_name, index = False)

	def detectBand(self, dens_df):
		"""
		Detect the band in the tube by evaluating the mean bacterial density across each row
		:return:
		"""
		dens_df = dens_df.T
		col_means = dens_df.values.mean(axis=0)

		#get the column with the maximum density
		max_dens_dx = np.where(col_means == np.amax(col_means))

		#get the location relative to the size of the tube
		max_dens_loc = max_dens_dx[0][0] * self.dx

		#update the list of band locations
		self.band_location.append(max_dens_loc)

	def step(self):
		self.schedule.step()
		self.stepConcentration()
		#self.bacteriaReproduce()
		#update the number of ticks which have occured
		self.ticks = self.ticks + 1
		print('TIME ELAPSED: '+ str(self.ticks*self.dt)+ ' seconds', flush = True)

def compute_concentration(model):
	return model.u






