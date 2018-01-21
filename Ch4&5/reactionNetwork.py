################################################################
# An implementation of reaction network. 
# Networks are simulated using the Gillespie SSA. 
################################################################
from random import random
import numpy as np
from copy import copy
from copy import deepcopy 
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sbn
sbn.set(rc={'image.cmap': 'cubehelix'})
################################################################
################# Encoding Arbitrary Networks ##################
#
# We introduce a structure which encodes an arbitrary 
# "protein" - interaction network. Networks are specified by a
# set of species which interact according to a set of reactions
# which have certain rates.  
# 
# Chemical reaction networks are simulated by the Gillespie SSA
################################################################
################################################################

class ReactionNetwork:
	############################################################
	# Inner members
	############################################################
	#
	# interactions - A list of possible chemical reactions in the system
	# Interactions are stored in a list [([a,b,c,...], [x,y,z,...]), ... ] where
	# ([a,b,c,...], [x,y,z,...]) encodes the reaction a+b+c+... -> x+y+z+... 
	# 
	# NOTE: By setting a=b and having the second list empty (for example) 
	# we can have reactions such as 2a->0
	# 
	# reaction_rates -  a list of rates for the reactions. We split the reactions and
	# rates for convenience/readability. 
	# 
	# numbers - The numbers of each chemical species in the system at the current time.
	# 
	# propensities - A list of functions f: numbers, rate -> reals which give propensities
	# for each reaction in interaction. We require len(propensities) = len(interactions)
	#
	# T - the time since the start of the current reaction network simulation.
	#
	# names - An optional argument assigning to each species a name. If not supplied the names will be
	# numerical from 0.
	#
	############################################################ 

	def __init__(self, interactions_list, reaction_rates, initial_numbers, reaction_propensities, names=None):

		assert len(interactions_list) == len(reaction_propensities)
		assert len(reaction_rates) == len(interactions_list)
		#assert len(names) == len(initial_numbers)

		#A list of reactions that can occur
		self.interactions = interactions_list
		#A list of rates for those reactions
		self.reaction_rates = reaction_rates
		#A list of species numbers for each reactant in the system
		self.numbers = initial_numbers
		#A list of propensity functions for the reactions
		self.propensities = reaction_propensities
		#The the current time
		self.T = 0
		#A history of the network
		self.history = [(self.T,copy(self.numbers))]
		#A history of reactions that have occurred
		self.reaction_history = []

		if names is not None:
			self.names = names
		else:
			self.names = map(str, range(len(initial_numbers)))


	def __str__(self):
		stringForm = ""
		j=0
		for r in self.interactions:
			s=""
			for i in r[0]:
				s = s+self.names[i]+" "
			s = s + "--" + str(self.reaction_rates[j]) + "--> "
			for i in r[1]:
				s= s+self.names[i]+" "
			stringForm = stringForm+s+"\n"
			j+=1

		return stringForm

	################################################################
	# Creates a plot of a number of runs of the stochastic resolution
	# for a specified species.
	################################################################
	def generateStochasticPlot(self, plotSpecies, number, c='k',Tlim = 10):
		# The trajectories to be plotted.
		sbn.set_context("talk")
		fig = plt.figure()
		ax = fig.add_subplot(111)

		trajectories = []
		for i in range(number):
			net = self.copy()
			net.resolve(Tlim)
			trajectories.append(net.history)

		# Helper function - Extracts the history of species i from a total history
		def extract(i,history):
			return map(lambda x: (x[0], x[1][i]), history)

		trajectoriesx = map(lambda y : extract(0,y), trajectories)
		trajectoriesy= map(lambda y : extract(1,y), trajectories)


		sbn.set_palette("husl")
		for i in range(len(trajectories)):
				t,p = zip(*trajectoriesx[i])
				plt.plot(range(len(t)),p,markersize=1 , lw=1.5, label='x')
			#t,p = zip(*trajectoriesy[i])
			#plt.plot(range(len(t)),p,markersize=1,c=sbn.xkcd_rgb["denim blue"], lw=3., label='y')

		#legend = ax.legend(loc='upper center', shadow=False)

		plt.xlabel("Number of Reactions")
		plt.ylabel("Number of Species x")
		plt.ylim((0,sum(self.numbers)))
		plt.xlim((0, max(map(len, trajectories))))
		plt.show()

	############################################################
	# Returns the number of species in the network
	############################################################
	def number_of_species(self):
		return len(self.numbers)
	############################################################
	# Returns the number of reactions in the network
	############################################################
	def number_of_reactions(self):
		return len(self.reaction_rates)

	############################################################
	# Returns the rho matrix from Cardelli's morphism paper
	#
	# rho(s,r) = number of s in reactants of reaction r
	############################################################
	def rho_matrix(self):

		#The number of times s appears in the reactants of r
		def rho(s,r):
			return sum(map(lambda x : x==s, r[0]))

		rhos = [[rho(s,r) for r in self.interactions] for s in range(self.number_of_species())]
		return np.matrix(rhos)

	############################################################
	# Returns the phi matrix  of instaneous stochiometries 
	# from Cardelli's morphisms paper
	#
	# phi(s,r) = r.rate(r.pi[s] - r.rho[s]) -- This is shorthand.
	############################################################
	def phi_matrix(self):

		#The (instantaneous) stochiometry of a single species s in reaction number n
		#NOTE: we use reaction number for convenience (to access the rates list more easily)
		def phi(s,n):
			rho_s = sum(map(lambda x : x==s,self.interactions[n][0]))
			pi_s  = sum(map(lambda x : x==s,self.interactions[n][1]))
			return self.reaction_rates[n]*(pi_s - rho_s)

		phis = [[phi(s,n) for n in range(self.number_of_reactions())] for s in range(self.number_of_species())]
		return np.matrix(phis)

	############################################################
	# Advances the reaction network by a single reaction
	############################################################
	def nextReaction(self):
		r1,r2 = random(), random()

		#Compute the propensity a_i(T) for each reaction
		a  = [self.propensities[i](self.numbers, self.reaction_rates[i]) for i in range(len(self.interactions))]

		a_0 = sum(a) 
		#Compute the tau step
		tau = (1/a_0)*np.log((1/r1))

		#Find the next reaction
		nextReaction=0
		lowerSum = 0
		upperSum = a[0]

		r2 = a_0*r2

		for j in range(0,len(a)):
			if r2>=lowerSum and r2<upperSum:
				nextReaction = j
				break
			lowerSum = upperSum
			upperSum += a[j+1]
		#Perform the reaction
		self.T += tau
		reaction = self.interactions[nextReaction]

		for k in reaction[0]:
			self.numbers[k] -= 1
		for k in reaction[1]:
			self.numbers[k] += 1

		self.history.append((self.T, copy(self.numbers)))
		self.reaction_history.append(nextReaction)

		return

	############################################################
	# Runs the network until no more reactions can occur. 
	# 
	# We call a network for which no more reactions can occur "resolved".
	############################################################
	def resolve(self,Tlim=-1.):
		resolved = self.is_resolved()
		while(not resolved):
			self.nextReaction()
			a   = [self.propensities[i](self.numbers, self.reaction_rates[i]) for i in range(len(self.interactions))]
			a_0 = sum(a)
			resolved = self.is_resolved()
			if Tlim > 0 and self.T > Tlim:
				resolved = True
		return 
	
	def is_resolved(self):
		a   = [self.propensities[i](self.numbers, self.reaction_rates[i]) for i in range(len(self.interactions))]
		a_0 = sum(a)
		return a_0 == 0


	############################################################
	# Finds the stationary distribution for the reaction network
	# using a Markov Chain method.
	#
	# method -  The method used to determine the distribution
	#			choose from 'explicit' (builds the markov 
	#			chain in full) or 'MCMC' (uses Markov Chain Monte
	#			Carlo)
	#
	# targetTest - A function which texts if a specific state is a target
	#			   for the distribution. We specify a function as many states
	#			   may correspond to the target. For example we might want x+b=0 
	#			   but have no interested in the value of x or b. 
	#
	#
	# Returns - A dictionary from states to probabilities representing
	# 			the probability that the chemical reaction progresses
	#			from that state to the target state (and stays there). This will be zero
	#			for all non-absorbing states 
	#
	# Note: For a large number of molecules this process is slow.
	# 
	# Warning: This method only terminates for reaction networks
	# which obey conservation of mass explicitely. The number of molecules in 
	# the system must remain constant.
	############################################################
	def getStationaryDistribution(self, targetTest, method='explicit'):
		import itertools
		import scipy
		import scipy.sparse as spr #Sparse matrix package
		from scipy.sparse.linalg import inv

		#First we build the state space split in two.
		# T - transient states 
		# A - absorbing states
		molList = [range(sum(self.numbers)+1) for i in range(self.number_of_species())]
		T,A = [],[]
		#For each possible collection of molecules
		for e in itertools.product(*molList):
			#If the total number is preserved
			if sum(e)==sum(self.numbers):
				tot_prob = 0.0
				s = list(e)
				#check is the state is absorbing and put in the appropriate list
				for k in range(len(self.interactions)):
					r = self.interactions[k]
					tot_prob += self.propensities[k](s, self.reaction_rates[k])
				if tot_prob > 0.0:
					T.append(s)
				else:
					A.append(s)

		# The state space has the following form:
		# S = [tttttttttttttttt...aaaaaa]
		# where t are transient states and a are absorbing states.
		#
		# The markov chain will take canonical form
		#
		# P = | Q R | Q = transient-transient transitions. R = transient to absorbing transitions
		#     | 0 I | 0 = the zero matrix. No absorbing->transient transitions possible. I = self-loops

		#Build the block matrices Q and R using sparse matrices
		#We build them from row stacks
		qstack, rstack = [],[] 

		#We store rows and copy them 
		qrow = spr.lil_matrix(np.zeros(len(T)))
		rrow = spr.lil_matrix(np.zeros(len(A))) 
		#Q = np.zeros((len(T), len(T)))
		#R = np.zeros((len(T), len(A)))
		for index in range(len(T)):
			t = T[index]

			tot_prob = 0.0
			to_add_listQ, to_add_listR = [],[]
			for k in range(len(self.interactions)):
				r = self.interactions[k]
				reaction_prob = self.propensities[k](t,self.reaction_rates[k])
				if reaction_prob > 0.:
					new_t = deepcopy(t)
					for reactant in r[0]:
						new_t[reactant] -= 1 
					for product in r[1]:
						new_t[product] += 1
					try:
						new_index = T.index(new_t)
						to_add_listQ.append((new_index, reaction_prob))

					except:
						new_index = A.index(new_t)
						to_add_listR.append((new_index, reaction_prob))

					tot_prob += reaction_prob

			row = deepcopy(qrow)
			for (new_index, prob) in to_add_listQ:
				row[0,new_index] = prob / tot_prob
			qstack.append(row)
			row = deepcopy(rrow)
			for (new_index,prob) in to_add_listR:
				row[0,new_index] = prob / tot_prob
			rstack.append(row)

		Q = spr.vstack(qstack, format='lil')
		R = spr.vstack(rstack, format='lil')

		#Build the funamental matrix
		N = spr.identity(len(T)) - Q
		N = inv(N) 

		#Build the distribution look up dictionary state->probs
		probs = np.array((N.dot(R)).todense())

		absorptionProbs = {}
		targets = [a for a in A if targetTest(a)]
		for t in T:
			prob = 0.0
			for a in targets:
				prob += probs[T.index(t), A.index(a)]
			absorptionProbs[tuple(t)] = prob

		for a in A:
			if targetTest(a):
				absorptionProbs[tuple(a)]=1.0
			else:
				absorptionProbs[tuple(a)]=0.0
		return absorptionProbs

	############################################################
	# Prints the network
	############################################################
	def printNetwork(self):
		j=0
		for r in self.interactions:
			s=""
			for i in r[0]:
				s = s+self.names[i]+" "
			s = s + "--" + str(self.reaction_rates[j]) + "--> "
			for i in r[1]:
				s= s+self.names[i]+" "
			print s
			j+=1

	############################################################
	# Returns a copy of the network
	############################################################
	def copy(self):
		ilist = deepcopy(self.interactions)
		rates = deepcopy(self.reaction_rates)
		nums  = deepcopy(self.numbers)

		return ReactionNetwork(ilist, rates, nums, self.propensities)

############################################################
############################################################
############################################################

############################################################
################ Approximate Majority ######################
#
# The n-species generalization of the Approximate Majority 
# network from Cardelli's paper. For equal starting populations
# this network will produce a population of all species k with 
# probability 1/k. 
#
# Rates are assumed to be 1.0 for all reactions.
#
############################################################
class ApproximateMajorityNetwork(ReactionNetwork): 

	############################################################
	# Builds a propensity function from the species involved.
	#
	# We use a seperate make function as anonymous functions
	# cannot be copied. In this way the function "propensity" is 
	# a different object each time make_propensity_function is called.
	############################################################
	def make_propensity_function(self,species_list):
			def propensity(x,y):
				return x[species_list[0]]*x[species_list[1]]*y
			return propensity

	def __init__(self, number_of_species, initial_numbers, rates=None, names=None):
		#Build the n-species reactions list

		interactions = []
		for i in range(number_of_species):
			for j in range(i+1,number_of_species):
				interactions.append(([i,j],[i,number_of_species]))
				interactions.append(([i,j],[j,number_of_species]))
		for i in range(number_of_species):
			interactions.append(([i,number_of_species],[i,i]))

		if not rates:
			rates = [1.0 for i in range(len(interactions))]

		#Build the propensity functions
		propensities = []
		for (r,p) in interactions:
			prop = self.make_propensity_function(r)
			propensities.append(prop)

		initial_numbers.append(0)

		ReactionNetwork.__init__(self, interactions, rates, initial_numbers, propensities, names)

############################################################
################ Direct Competition ########################
#
# Builds the DC switch (Fig 4b from Cell Cycle Switch compute AM)
#
############################################################
class DirectCompetition(ReactionNetwork):

	############################################################
	# Builds a propensity function from the species involved.
	#
	# We use a seperate make function as anonymous functions
	# cannot be copied. In this way the function "propensity" is 
	# a different object each time make_propensity_function is called.
	############################################################
	def make_propensity_function(self,species_list):
			def propensity(x,y):
				return x[species_list[0]]*x[species_list[1]]*y
			return propensity

	def __init__(self, reaction_rates, initial_numbers, names=['x','y']):


		interactions = []
		interactions.append(([0,1],[0,0])) #x+y -> x+x
		interactions.append(([0,1],[1,1])) #x+y -> y+y
		propensities = []
		for (r,p) in interactions:
			prop = self.make_propensity_function(r)
			propensities.append(prop)

		ReactionNetwork.__init__(self, interactions, reaction_rates, initial_numbers, propensities, names)

	def DC_hedge(self,g,g_max,s1=1.,s2=1.):
		x0 = g
		y0 = g_max-g

		p = s1/(s1+s2)
		q = s2/(s1+s2)
		if p==q:
			return float(x0) / (x0 + y0)
		else:
			prob = (1.-(q/p)**x0) / (1. - (q/p)**(x0+y0))
			return prob

	def get_DC_hedge(self,s1,s2):
		def dch(g):
			return self.DC_hedge(g,sum(self.numbers), s1,s2)
		return dch

	def getStationaryDistribution(self):
		return self.get_DC_hedge(self.reaction_rates[0], self.reaction_rates[1])


############################################################
################ Direct Competition ########################
#
# Builds the DC switch (Fig 4b from Cell Cycle Switch compute AM)
#
############################################################
class AMDCHybrid(ReactionNetwork):

	############################################################
	# Builds a propensity function from the species involved.
	#
	# We use a seperate make function as anonymous functions
	# cannot be copied. In this way the function "propensity" is 
	# a different object each time make_propensity_function is called.
	############################################################
	def make_propensity_function(self,species_list):
			def propensity(x,y):
				return x[species_list[0]]*x[species_list[1]]*y
			return propensity

	def __init__(self, reaction_rates, initial_numbers, names=['x','y','b']):


		interactions = []
		interactions.append(([0,1],[0,0])) #x+y -> x+x
		interactions.append(([0,1],[1,1])) #x+y -> x+y
		interactions.append(([0,1],[0,2])) #x+y -> x+b
		interactions.append(([0,1],[1,2])) #x+y -> y+b
		interactions.append(([0,2],[0,0])) #x+b -> x+x
		interactions.append(([1,2],[1,1])) #y+b -> x+y

		propensities = []
		for (r,p) in interactions:
			prop = self.make_propensity_function(r)
			propensities.append(prop)

		ReactionNetwork.__init__(self, interactions, reaction_rates, initial_numbers, propensities, names)

# net = DirectCompetition([1.0,1.0], [50,50])
# print net.getStationaryDistribution()

############################################################
################ Processive Modification ###################
#
# Builds the PM switch (Fig 4b from Cell Cycle Switch compute AM)
#
############################################################
class ProcessiveModification(ReactionNetwork):

	############################################################
	# Builds a propensity function from the species involved.
	#
	# We use a seperate make function as anonymous functions
	# cannot be copied. In this way the function "propensity" is 
	# a different object each time make_propensity_function is called.
	############################################################
	def make_propensity_function(self,species_list):
			def propensity(x,y):
				return x[species_list[0]]*x[species_list[1]]*y
			return propensity

	def __init__(self, reaction_rates, initial_numbers, names=['x','y','b','c']):


		interactions = []
		interactions.append(([0,1],[1,2])) #x+y -> y+b
		interactions.append(([1,0],[0,3])) #y+x -> x+c
		interactions.append(([3,0],[0,0])) #c+x -> x+x
		interactions.append(([2,1],[1,1])) #b+y -> y+y

		propensities = []
		for (r,p) in interactions:
			prop = self.make_propensity_function(r)
			propensities.append(prop)

		ReactionNetwork.__init__(self, interactions, reaction_rates, initial_numbers, propensities, names)

############################################################
##################### Dimerization #########################
#
# Builds the Dimerization switch
# (Fig 4b from Cell Cycle Switch compute AM)
#
############################################################
class Dimerization(ReactionNetwork):
	def make_propensity_function(self,species_list):
			def propensity(x,y):
				if species_list[0]==species_list[1]:
					return x[species_list[0]]*(x[species_list[1]]-1)*y
				else: 
					return x[species_list[0]]*x[species_list[1]]*y
			return propensity

	def __init__(self, reaction_rates, initial_numbers, names=['x','y','b','c']):

		interactions = []
		interactions.append(([0,0],[2])) #x+x -> b
		interactions.append(([1,1],[3])) #y+y -> c
		interactions.append(([1,2],[2,0]))
		interactions.append(([0,3],[3,1]))
		
		propensities = []
		for (r,p) in interactions:
			prop = self.make_propensity_function(r)
			propensities.append(prop)

		ReactionNetwork.__init__(self, interactions, reaction_rates, initial_numbers, propensities, names)


############################################################
################ Simply Catalytic ##########################
#
# Builds the SC Switch (Fig 4b from Cell Cycle Switch compute AM)
############################################################
class SimplyCatalytic(ReactionNetwork):
	############################################################
	# Builds a propensity function from the species involved.
	#
	# We use a seperate make function as anonymous functions
	# cannot be copied. In this way the function "propensity" is 
	# a different object each time make_propensity_function is called.
	############################################################
	def make_propensity_function(self,species_list):
			def propensity(x,y):
				return x[species_list[0]]*x[species_list[1]]*y
			return propensity

	def __init__(self, reaction_rates, initial_numbers, names=['x','y','b','p','q','r','s','t','u','w','z']):
		interactions = []
		interactions.append(([0,9],[9,2])) # x+w -> w+b
		interactions.append(([2,9],[9,1])) # b+w -> w+y
		interactions.append(([1,5],[5,2])) # y+r -> r+b
		interactions.append(([2,5],[5,0])) # b+r -> r+x
		interactions.append(([9,6],[6,8])) # w+s -> s+u
		interactions.append(([8,6],[6,10])) # u+s -> s+z
		interactions.append(([10,1],[1,8])) # z+y -> y+u
		interactions.append(([8,1],[1,9])) # u+y -> y+w
		interactions.append(([3,0],[0,4])) # p+x -> x+q
		interactions.append(([4,0],[0,5])) # q+x -> x+r
		interactions.append(([5,7],[7,4])) # r+t -> t+q
		interactions.append(([4,7],[7,3])) # q+t -> t+p

		propensities = []
		for (r,p) in interactions:
			prop = self.make_propensity_function(r)
			propensities.append(prop)

		ReactionNetwork.__init__(self, interactions, reaction_rates, initial_numbers, propensities, names)

############################################################
####### Reaction Network with Environmental Input ##########
#
# This builds a ReactionNetwork by introducing a set of 
# environmental factors which are taken up by the network 
# at specifiable rates.
#
# These outside species are introduced into a system through
# a set of first order reactions 0--k-->A. The rate k could be
# for example limited by a cell's maximal update rate.
#
# Builds a ReactionNetwork according to:
# interactions_list, reaction_rates, initial_numbers, reaction_propensities
# 
# adding additional production reactions for each species in field_species at
# the corresponding field_rate.
############################################################
class EnvironmentalNetwork(ReactionNetwork):

	def __make_zero_order_propensity(self):
		return lambda x,y : y

	def __init__(self, interactions_list, reaction_rates, initial_numbers, reaction_propensities, field_species, field_rates):
		for i in range(len(field_species)):
			interactions_list.append(([], [field_species[i]]))
			reaction_rates.append(field_rates[i])
			reaction_propensities.append(self.__make_zero_order_propensity())

		ReactionNetwork.__init__(self, interactions_list, reaction_rates, initial_numbers, reaction_propensities)

	def is_resolved(self):
		a   = [self.propensities[i](self.numbers, self.reaction_rates[i]) for i in range(len(self.interactions))]
		a_0 = sum(map(lambda x : x>0, a))
		return a_0 == 1
