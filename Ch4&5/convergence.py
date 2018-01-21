###################################################################
# Computees the invasion probability for a mutant bet-hedging in an
# existing population.
#
# Daniel Nichol 19/06/2015
###################################################################
import sys

import numpy as np
import scipy as sp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib as mpl

import reactionNetwork as rn
import hedges
import invasion as inv
from invasion import find_invasion_prob

import seaborn as sbn
sbn.set(rc={'image.cmap': 'cubehelix'})
###################################################################
# Parameters
###################################################################
w = [2.0, 1.01] #Section 4.5.3
g_max = 60
DC_hedge = hedges.DC_hedge
DC_hedge_x = hedges.get_DC_hedge(1.1,1.0)
DC_hedge_y = hedges.get_DC_hedge(1.0,1.1)
AM_hedge = hedges.get_AM_hedge(60)

###################################################################
# Defining a genotype space
#
# Individuals are determined by a genotype g in the set {0,1,...,g_max}.
# Mutations are determined by g->g-1 or g->g+1. If g is 0 or g_max then
# then only one mutation can occur. This mutations only occurs 50% of the time
#
# (Equation 4.2)
###################################################################
class GenotypeSpace:
    def __init__(self, g_max):
        self.g_max = g_max

    def get_gmax(self):
        return self.g_max

    #Returns a mutation of the genotype g
    def mutate(self,g):     
        r = np.random.random()
        if g==self.g_max:
            if r<0.5:
                return g-1
            return g-1
        elif g==0:
            if r<0.5:
                return g+1
        else:
            if r<0.5:
                return g+1
            else:
                return g-1

g_space = GenotypeSpace(60)
####################################################################
# Calculates an expected convergence time
####################################################################
def expected_convergence_time(g,f,gp_map, g_max=60):
    e = 0.
    for g_new in range(g+1,g_max+1):
        inv_prob = 0.5*find_invasion_prob(gp_map(g,g_max), gp_map(g_new, g_max), f)
        e += 1./inv_prob
        g = g_new
    return e

####################################################################
# Stochastically simulates convergence and returns a convergence time
#
# (Figure 4.7)
####################################################################
def get_convergence_time(g_0, g_space, end_hedge, gp_map, f):
    resident_hedge = gp_map(g_0,g_space.get_gmax())
    t = 0
    g = g_0
    while np.abs(resident_hedge - end_hedge) > 10**-30:
        new_g = g_space.mutate(g)
        invader_hedge = gp_map(new_g, g_space.get_gmax())
        r = np.random.random()
        fix_p = find_invasion_prob(resident_hedge, invader_hedge, f)
        if r < fix_p:
            g = new_g
            resident_hedge = invader_hedge
        t+=1
        if t % 5000 == 0:
            print t
    return t

####################################################################
# This is the code shown in Figure 4.7 but with extra return values
# (the curves trajectories through p- and g-space.)
#
# Generate a stochastic run of convergence (Data for Figure 4.8)
####################################################################
def stoch_curve(g_0, end_hedge, gp_map, f):
    g_space = GenotypeSpace(g_max)
    resident_hedge = gp_map(g_0, g_max)
    hedge_curve = [resident_hedge]
    g_curve = [g_0]
    g = g_0
    num_events=0
    # each loop corresponds to a mutation
    while np.abs(resident_hedge - end_hedge) > 10**-20 and num_events<100000:
        num_events+=1
        new_g = g_space.mutate(g)
        invader_hedge = gp_map(new_g, g_max)
        r = np.random.random()
        if invader_hedge < resident_hedge:
            fix_p = 0.0
        else:
            fix_p = find_invasion_prob(resident_hedge, invader_hedge, f)
        if r < fix_p:
            g = new_g
            resident_hedge = invader_hedge
        hedge_curve.append(gp_map(g,g_max))
        g_curve.append(g)
    print num_events #This is t_conv in Fig 4.7
    return hedge_curve, g_curve

####################################################################
# Generates a number of stochastic convergence-curves and saves them
# at './Convergence'. 
####################################################################
def convergence_curves(f, trials):
    try: 
        os.makedirs('./Convergence')
    except OSError:
        pass #It already exists

    names = ["DC","DCx","DCy","AM"]
    gp_map_list = [DC_hedge, DC_hedge_x, DC_hedge_y, AM_hedge]
    gs = [30,7,53,30]
    #for each map interest
    for j in range(len(gp_map_list)):
        cvs, gcvs = [],[]
        for i in range(trials):
            print "Trial: ", i
            hcv, gcv = stoch_curve(gs[j], 1.0, gp_map_list[j], f)
            cvs.append(hcv)
            gcvs.append(gcv)
        np.save('./Convergence/'+names[j]+'_h', cvs)
        np.save('./Convergence/'+names[j]+'_g', gcvs)

####################################################################
# Plots the evolutionary trajetories through genotype space
# generated above. The plots are compiled to form Figure 4.8
####################################################################
def convergence_plot_g(name, logscale=False):
    cvs = np.load('./Convergence/'+name+'_g.npy')

    expect_conv_DC = expected_convergence_time(30, [2.0,1.01], DC_hedge, 60)
    expect_conv_DCx = expected_convergence_time(7, [2.0,1.01], DC_hedge_x, 60)
    expect_conv_DCy = expected_convergence_time(53, [2.0,1.01], DC_hedge_y, 60)

    fig, ax = plt.subplots()
    plt.ylabel("Genotype"+r'$x_0$')
    plt.xlabel("Mutational Events")
    colpal = sbn.color_palette("Spectral", 30)

    if name=="DCy":
        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=5)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        for i in range(len(cvs)):
            cv = cvs[i]
            cv = cv[1:]
            plt.plot(cv, c = colpal[i])

        plt.ylim(0,60)

    if name=="DC":

        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=5)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DC, color=sbn.xkcd_rgb["pale red"], linewidth=5)
        ax.annotate("DC Expected Convergence", xy=(expect_conv_DC, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)
            
        for i in range(len(cvs)):
            cv = cvs[i]
            cv = cv[1:]
            plt.plot(cv, c = colpal[i])
        plt.ylim(0,60)

    if name=="DCx":

        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=2)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DC, color=sbn.xkcd_rgb["pale red"], linewidth=2)
        ax.annotate("DC Expected Convergence", xy=(expect_conv_DC, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DCx, color=sbn.xkcd_rgb["denim blue"], linewidth=2)
        ax.annotate("DCx Expected Convergence", xy=(expect_conv_DCx, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        for i in range(len(cvs)):
            cv = cvs[i]
            cv = cv[1:]
            plt.plot(cv, c = colpal[i])

        plt.ylim(0,60)

    if name=="AM":

        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=2)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DC, color=sbn.xkcd_rgb["pale red"], linewidth=2)
        ax.annotate("DC Expected Convergence", xy=(expect_conv_DC, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DCx, color=sbn.xkcd_rgb["denim blue"], linewidth=2)
        ax.annotate("DCx Expected Convergence", xy=(expect_conv_DCx, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        for i in range(len(cvs)):
            cv = cvs[i]
            plt.plot(cv, c = colpal[i])
        plt.ylim(0,60)

    plt.title("Convergence to Non-Hedging Strategy "+name)
    plt.show()


def convergence_plot(name, logscale=False):
    cvs = np.load('./Convergence/'+name+'_h.npy')
    fig, ax = plt.subplots()
    plt.ylabel("Probability of Phenotype B")
    plt.xlabel("Mutational Events")
    colpal = sbn.color_palette("Spectral", 30)

    expect_conv_DC = expected_convergence_time(30, [2.0,1.01], DC_hedge, 60)
    expect_conv_DCx = expected_convergence_time(7, [2.0,1.01], DC_hedge_x, 60)
    expect_conv_DCy = expected_convergence_time(53, [2.0,1.01], DC_hedge_y, 60)

    if name=="DCy":

        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=5)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        for i in range(len(cvs)):
            cv = cvs[i]
            cv = cv[1:]
            cv = map(lambda x : 1.-x, cv)
            plt.plot(cv, c = colpal[i])

        plt.ylim(0.0,1.0)

    if name=="DC":

        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=5)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DC, color=sbn.xkcd_rgb["pale red"], linewidth=5)
        ax.annotate("DC Expected Convergence", xy=(expect_conv_DC, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)
            
        for i in range(len(cvs)):
            cv = cvs[i]
            cv = cv[1:]
            cv = map(lambda x : 1.-x, cv)
            plt.plot(cv, c = colpal[i])
        plt.ylim(0.0,1.0)

    if name=="DCx":

        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=2)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DC, color=sbn.xkcd_rgb["pale red"], linewidth=2)
        ax.annotate("DC Expected Convergence", xy=(expect_conv_DC, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DCx, color=sbn.xkcd_rgb["denim blue"], linewidth=2)
        ax.annotate("DCx Expected Convergence", xy=(expect_conv_DCx, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        for i in range(len(cvs)):
            cv = cvs[i]
            cv = cv[1:]
            cv = map(lambda x : 1.-x, cv)
            plt.plot(cv, c = colpal[i])

        plt.ylim(0.0,0.01)
        plt.xlim(0, 100000)

    if name=="AM":

        ax.axvline(expect_conv_DCy, color=sbn.xkcd_rgb["medium green"], linewidth=2)
        ax.annotate("DCy Expected Convergence", xy=(expect_conv_DCy, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DC, color=sbn.xkcd_rgb["pale red"], linewidth=2)
        ax.annotate("DC Expected Convergence", xy=(expect_conv_DC, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        ax.axvline(expect_conv_DCx, color=sbn.xkcd_rgb["denim blue"], linewidth=2)
        ax.annotate("DCx Expected Convergence", xy=(expect_conv_DCx, 1.23), xytext=(2.0, -150),
                xycoords=('data', 'axes fraction'), textcoords='offset points',
                horizontalalignment='left', verticalalignment='center', rotation=270)

        plt.ylim(10**-6,1.0)
        for i in range(len(cvs)):
            cv = cvs[i]
            cv = map(lambda x : 1.-x, cv)
            plt.plot(cv, c = colpal[i])
        plt.yscale('log')


    plt.title("Convergence to Non-Hedging Strategy "+name)
    plt.show()


###################################################################
# Returns the expected genotype after a given number of mutational events.
#
# This method is represented in Figure 4.9 in the green dashed box.
###################################################################
def expectation_of_hedge(g,f,gp_map,t):
    curr_t = 0
    while curr_t < t:
        g_new = g+1
        inv_prob = 0.5*find_invasion_prob(gp_map(g,g_max), gp_map(g_new, g_max), f)
        e = 1. / inv_prob
        if curr_t + e < t:
            g = g_new
            curr_t += e
        else:
            break
        if g == g_max:
            return g_max
    return g