###################################################################
# Computees the invasion probability for a mutant bet-hedging in an
# existing population.
#
# Daniel Nichol 19/06/2015
###################################################################
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
from scipy.optimize import fsolve
import seaborn as sbn
sbn.set(rc={'image.cmap': 'cubehelix'})

import reactionNetwork as rn
import hedges

###################################################################
# Parameters
###################################################################
w = [2.0, 1.01] #Section 4.5.3

####################################################################
# Population and invasion dynamics
####################################################################

###################################################################
# Returns the transition matrix for the population dynamics 
###################################################################
def get_M(hedge, w_vector, current_avg = 1.0):
	M = np.zeros((2,2))
	M[0,0] = w_vector[0]*hedge / current_avg 
	M[0,1] = w_vector[0]*(1.-hedge) / current_avg
	M[1,0] = w_vector[1]*hedge / current_avg
	M[1,1] = w_vector[1]*(1.-hedge) / current_avg

	return np.matrix(M)

###################################################################
# Returns the spectral radius of M
#
# This value is also the asymptotic growth rate (or avg fitness)
# lim t->inf |n(t+1)|_1 / |n(t)|_1
#
# As well as the unit sum eigenvector v associated with 
# this radius. This eigenvector gives the stationary population
# distribution as t->inf
#
# v = lim t-> inf n(t) / |n(t)|_1
####################################################################
def get_lambda_v(M):
	evals, evecs = np.linalg.eig(M)[0], np.linalg.eig(M)[1]
	spec_rad = max(evals)
	i = list(evals).index(spec_rad)
	perron_v = evecs[:,i]
	perron_v = perron_v / sum(perron_v)

	return spec_rad, perron_v

####################################################################
# Gets the avg fitness of a hedging population at steady state
####################################################################
def get_avg_fitness(hedge, fitness):
	P = get_M(hedge, fitness).T
	spec_rad, perron_v = get_lambda_v(P)
	#normalize the population distribution vector, perron_v
	tot = sum(perron_v)
	perron_v  = perron_v / tot
	avg_fitness = fitness[0]*perron_v[0,0] + fitness[1]*perron_v[1,0]

	return avg_fitness

####################################################################
# Invasion probability
#
# The invasion probability is calculated by numerically solving
# Equation 4.12 and substituting the result into Equation 4.12
####################################################################
def find_invasion_prob(resident_hedge, invader_hedge, fitness):
	resident_avg = get_avg_fitness(resident_hedge, fitness)
	M = get_M(invader_hedge, fitness, current_avg = resident_avg)

	def func(x): #Eqn 4.13
		out = [1-x[0]-np.exp( -(M[0,0]*x[0] + M[0,1]*x[1]) )]
		out.append(1-x[1]-np.exp( -(M[1,0]*(x[0]) + M[1,1]*x[1]) ))

		return out

	from scipy.optimize import root
	pi1, pi2 = root(func, [0.5,0.5], method='hybr').x
	if pi1 < 0.:
		pi1 = 0.
	if pi2 < 0.:
		pi2 = 0.

	if pi1 > 1. or pi2 > 1.:
		print "!"
	return invader_hedge*pi1 + (1-invader_hedge)*pi2 #Eqn 4.12

####################################################################
# Generating the figures
####################################################################

####################################################################
# PIP plot for a resident hedge p1 and an invader p2.
####################################################################
def pairwise_invasion_plot(fitness):
	#build the data
	x = np.arange(0.0,1.0,0.01)
	y = np.arange(0.0,1.0,0.01)
	pip = [[find_invasion_prob(p1,p2,fitness) for p2 in x] for p1 in y]

	#set up
	fig = plt.figure()
	ax = fig.add_subplot(111)
	seamap = sbn.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
	ax.xaxis.set_ticks_position('bottom')

	#plot
	cax = ax.imshow(pip, interpolation='spline16', origin='lower', vmax=1.0, cmap="BuGn")
	fig.colorbar(cax)

	#contours
	cset = plt.contour(pip,np.arange(0.1,1.,0.1),linewidths=.2,colors='k')
	plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)

	plt.xticks(range(0,101,10), map(str, np.arange(0.0,1.01,0.1)))
	plt.yticks(range(0,101,10), map(str, np.arange(0.0,1.01,0.1)))

	plt.xlabel("Invader Hedge, p2")
	plt.ylabel("Resident Hedge, p1")
	plt.title("Probability of Invasion")
	plt.show()

####################################################################
# Figure 4.6
####################################################################

#4.6(A)
def g_vs_avgf_curve(g_max, w, gp_map_list, title_list, colors, linestyles=['-', '--', '-.',':']):
	domain = [g for g in range(g_max+1)]
	crvs = []

	for gp_map in gp_map_list:
		ps = map(lambda x : gp_map(x,g_max), domain)
		crvs.append(ps)

	f1s = [map(lambda x : get_avg_fitness(x,w), c) for c in crvs]

	_,ax = plt.subplots()
	for i in range(len(f1s)):
		plt.plot(domain,f1s[i], label=title_list[i],c=colors[i], ls=linestyles[i], lw=2.0)

	plt.legend(bbox_to_anchor=(0., -0.2, 1., .102), loc=4,
           ncol=len(gp_map_list), mode="expand", borderaxespad=0.)

	plt.title("Genotype vs Average Population Fitness")
	plt.ylabel("Average Population Fitness")
	plt.xlabel('Genotype '+r'$g$')
	plt.grid('on')
	plt.subplots_adjust(bottom=0.3)
	plt.show()

#4.6(B)
def diminishing_returns_plot(w, g_max, gp_map_list, title_list, colors, linestyles = ['-', '--', '-.',':']):
	domain = [g for g in range(g_max)]
	crvs = []
	for gp_map in gp_map_list:
		crvs.append(map(lambda x : find_invasion_prob(gp_map(x,g_max),gp_map(x+1,g_max),w), domain))
	for i in range(len(gp_map_list)):
		plt.plot(domain,crvs[i],label=title_list[i], c=colors[i], ls = linestyles[i], lw=2.0)

	plt.ylim((0.0,0.25))
	plt.xlim((0,g_max-1))
	plt.title('Fixation Probabilities for Successive Beneficial Mutations')
	plt.ylabel('Invasion probability for '+r'$g+1$')
	plt.xlabel('Genotype '+r'$g$')
	plt.legend(bbox_to_anchor=(0., -0.2, 1., .102), loc=4,
       ncol=len(gp_map_list), mode="expand", borderaxespad=0.)

	plt.show()

#4.6(C)
def genotype_invasion_plot(gp_map,fitness, g_max, hedgename=''):

	sbn.set_style("white")
	def in_prob(g_res,g_inv):
		p1 = gp_map(g_res, g_max)
		p2 = gp_map(g_inv, g_max)
		return find_invasion_prob(p1,p2,fitness)

	pip = [[in_prob(g1,g2) for g2 in range(g_max+1)] for g1 in range(g_max+1)]

	#set up
	fig = plt.figure()
	ax = fig.add_subplot(111)

	#plot
	cax = ax.matshow(pip, interpolation='none', origin='lower', vmax=1.0, cmap="BuGn")
	fig.colorbar(cax)

	#contours
	cset = plt.contour(pip,np.arange(0.1,1.,0.1),linewidths=.2,colors='k')
	plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)

	ax.xaxis.set_ticks_position('bottom')

	plt.xlabel("Invader Genotype, "+r'$x^{-}_0$')
	plt.ylabel("Resident Genotype, "+r'$x_0$')
	plt.title("Probability of Invasion, "+hedgename)

	plt.show()

###################################################################
# Altering rates figures (Robustness to Parameter variation)
##################################################################
def rates_figure(gp_builder, title):
	from am_gpmap_lookup import get_AM_lookup

	min, max = (0, 2.0)
	step = 0.1

	# Setting up a colormap that's a simple transtion
	mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue','red'])

	# Using contourf to provide my colorbar info, then clearing the figure
	Z = [[0,0],[0,0]]
	levels = np.arange(min,max+step,step)
	CS3 = plt.contourf(Z, levels, cmap=mymap)
	plt.clf()

	for r1 in np.arange(0.1,2.01,0.1):
		dch = gp_builder(r1)

		tmp = [dch(g,g_max) for g in range(g_max+1)]
		r = r1/2.
		g = 0
		b = 1-r
		plt.plot(tmp,color=(r,g,b))
	cb = plt.colorbar(CS3)
	cb.set_label('r2')
	cb.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0])
	plt.xlabel('Genotype')
	plt.ylabel('Probability of Phenotype A')
	plt.title(title)
	plt.show()


####################################################################
# Parameter Sensitivity (Appendix)
####################################################################
def gmax_figure():
	from matplotlib import rc
	def get_start(g_max, hedge):
		best_y = 0
		best = 1.0
		for y in range(0,g_max):
			if np.abs(hedge(y, g_max) - 0.5) < best:
				best = np.abs(hedge(y, g_max) - 0.5)
				best_y = y
		return best_y


	gmsam = range(3,21,2)
	gmsx = range(0,76,5)
	gms = range(10,201,10)
	dc_convs = map(lambda x : expected_convergence_time(get_start(x, DC_hedge), w, DC_hedge, g_max = x), gms)
	dcx_convs = map(lambda x : expected_convergence_time(get_start(x, DC_hedge_x), w, DC_hedge_x, g_max = x), gmsx)
	dcy_convs = map(lambda x : expected_convergence_time(get_start(x, DC_hedge_y), w, DC_hedge_y, g_max = x), gms)

	am_convs = []

	from am_gpmap_lookup import get_AM_lookup

	for gm in  gmsam:
		look_up = get_AM_lookup(gm,1.,1.,1.,1.)
		def AM_hedge(g,g_max):
			return look_up[g][g_max-g]

		am_convs.append(expected_convergence_time(get_start(gm, AM_hedge), w, AM_hedge, g_max = gm))

	plt.plot(gms, dc_convs, c=sbn.xkcd_rgb["pale red"], label = "DC")
	plt.plot(gmsx, dcx_convs, c=sbn.xkcd_rgb["denim blue"], linestyle='--', label = "DCx")
	plt.plot(gms, dcy_convs, c=sbn.xkcd_rgb["medium green"], ls= '-.', label = "DCy")
	plt.plot(gmsam, am_convs, c='k', ls= ':', label="AM")

	plt.ylim(0,50000)
	plt.ylabel(r'Convergence Time (Mutational Events)')
	plt.xlabel(r'$g_{max}$')
	plt.title(r'Convergence Times for Varying $g_{max}$')


	plt.legend(bbox_to_anchor=(0., -0.2, 1., .102), loc=4,
		ncol= 4 , mode="expand", borderaxespad=0.)

	plt.show()

def varying_WA():
	from matplotlib import rc
	W_As = np.arange(2.0, 10.02, 0.05)
	dc_convs = map(lambda wa : expected_convergence_time(30, [wa,1.01], DC_hedge, 60), W_As)
	dcx_convs = map(lambda wa : expected_convergence_time(7, [wa,1.01], DC_hedge_x, 60), W_As)
	dcy_convs = map(lambda wa : expected_convergence_time(53, [wa,1.01], DC_hedge_y, 60), W_As)

	plt.plot(W_As, dc_convs, c=sbn.xkcd_rgb["pale red"], label = "DC")
	plt.plot(W_As, dcx_convs, c=sbn.xkcd_rgb["denim blue"], linestyle='--', label = "DCx")
	plt.plot(W_As, dcy_convs, c=sbn.xkcd_rgb["medium green"], ls= '-.', label = "DCy")

	plt.legend(bbox_to_anchor=(0., -0.2, 1., .102), loc=4,
		ncol= 3 , mode="expand", borderaxespad=0.)

	plt.ylabel(r'Convergence Time (Mutational Events)')
	plt.xlabel(r'$w_A$')
	plt.title(r'Convergence Times for Increasing $w_A$')

	plt.show()
