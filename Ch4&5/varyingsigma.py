#######################################################
# Implements the methods used to exp
#######################################################
from copy import deepcopy

import cmocean
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.linalg as la
import scipy as sp
import seaborn as sns
sns.set_context("paper")
sns.set_style("darkgrid")

import hedges as hds
from memmodel import *

####################################################### 
# For all sigma values in the specified range, explores
# the full space of metronomic therapies and records the
# optimal therapy and the maximum dose density therapy
# growth rates 
####################################################### 
def vary_sig(hedge, name, xb, interval_len = 7):
    print hedge, name
    num_days = 14
    for sk in [3,4,5,6,7,8]:
        sig = 6.77 * 10**(-sk)
        M_drug = get_memory_M(sig, d_A_drug, d_B_drug, b_A, b_B, xb, 60-xb, hedge)**(60*60*24)
        M_nodrug = get_memory_M(sig, d_A, d_B, b_A, b_B, xb, 60-xb, hedge)**(60*60*24)

        best_rhos = [np.infty for i in range(num_days+1)]
        best_strats = [[] for i in range(num_days+1)]
        all_rhos = [[] for i in range(num_days+1)]
        mdd_rhos = [0. for i in range(num_days+1)]

        for k in range(0, 2**num_days):
            print k, "/", 2**num_days
            strat = map(int, "{0:b}".format(k).zfill(num_days))  

            Mcycle = np.identity(M_drug.shape[0])
            for j in strat:
                if j == 1:
                    Mcycle = (M_drug** interval_len) * Mcycle 
                else:
                    Mcycle = (M_nodrug ** interval_len) * Mcycle 

            rho, _ = spectral_rad(Mcycle)
            rho = np.real(rho)
            cycle_f = (1./14)*np.log(rho) 

            dose = sum(strat)
            all_rhos[dose].append(cycle_f)

            if "{0:b}".format(k).zfill(num_days) == "".join(['1' for i in range(dose)]).ljust(num_days, '0'):
                mdd_rhos[dose] = cycle_f

            if cycle_f < best_rhos[dose]:
                best_rhos[dose] = cycle_f
                best_strats[dose] = strat

        np.save('./VaryingSigma/best_rhos_'+str(sk)+'_'+str(num_days).zfill(3)+'_'+name+str(xb), best_rhos)
        np.save('./VaryingSigma/best_strats_'+str(sk)+'_'+str(num_days).zfill(3)+'_'+name+str(xb), best_strats)
        np.save('./VaryingSigma/all_rhos_'+str(sk)+'_'+str(num_days).zfill(3)+'_'+name+str(xb), all_rhos)
        np.save('./VaryingSigma/mdd_rhos_'+str(sk)+'_'+str(num_days).zfill(3)+'_'+name+str(xb), mdd_rhos)

#######################################################
# Generates the full data set for Section 5.7
#######################################################
def generate_all_sv(interval_len = 7):
    vary_sig(DC_hedge, "DC", 59, interval_len)
    vary_sig(DCy_hedge, "DCy", 59, interval_len)
    vary_sig(DCx_hedge, "DCx", 46, interval_len)
    vary_sig(AM_hedge, "AM", 39, interval_len)

    vary_sig(DC_hedge, "DC", 30, interval_len)
    vary_sig(DCy_hedge, "DCy", 53, interval_len)
    vary_sig(DCx_hedge, "DCx", 7, interval_len)
    vary_sig(AM_hedge, "AM", 30, interval_len)

#######################################################
# Returns the growth rate for a given strategy for two 
# specified pop matrices. Note xb and hedge are used to
# determine an initial conditions.
#
# This function is used by sigma_rho_analysis to build
# the curved in Figs 5.19 and 5.20
#######################################################
def get_growth_curve(xb, hedge, strat, cycles, M_drug, M_nodrug):
    #The initial population:    
    yb = 60-xb
    M_max = 2*(xb+yb)+1
    init_pop = [0. for i in range(4*(xb+yb)+2)]
    init_pop[xb+yb] = hedge(xb,yb) * 1.0
    init_pop[xb+yb+M_max] = (1-hedge(xb,yb))*1.0
    init_pop = np.matrix(init_pop).T
    #Find the stationary (no drug) distribution.
    for t in range(7):
        init_pop = M_nodrug * init_pop
    init_pop = init_pop / np.sum(init_pop)
    pop = init_pop
    cv = []
    T_end = len(strat)*cycles
    for t in range(T_end):
        for hour in range(24):
            cv.append(np.sum(pop))
            if strat[t%len(strat)]:
                pop = M_drug * pop
            else:
                pop = M_nodrug * pop
    return cv

#######################################################
# Builds the curves linking sigma to the growth
# rate for the mdd and best metronomic therapies.
#######################################################
def sigma_rho_analysis(gs_num, dose, interval_len = 7):
    num_days = 14
    cycles = 4
    sks = [8,7,6,5,4,3]
    best_strats, best_rhos = [[],[],[],[]], [[],[],[],[]]
    #Extract the strats.
    for hn in range(len(hedges)):
        xb = gs_sets[1][hn]
        for sk in sks:
            strats = np.load('./VaryingSigma/best_strats_'+str(sk)+'_'+str(num_days).zfill(3)+'_'+hedge_names[hn]+str(xb)+'.npy')
            rhos = np.load('./VaryingSigma/best_rhos_'+str(sk)+'_'+str(num_days).zfill(3)+'_'+hedge_names[hn]+str(xb)+'.npy')
            best_strats[hn].append(strats[dose])
            best_rhos[hn].append(rhos[dose])

    mdd = [1 for i in range(dose)] + [0 for i in range(num_days - dose)] 
      
    #Build the dists.
    for hn in range(len(hedges)):
        xb = gs_sets[gs_num][hn]
        sk_best_cv = []
        for skn, sk in enumerate(sks):
            sig = 6.77 * 10**(-sk)
            M_drug, M_no_drug = get_one_hour_matrices(sig, xb, hedges[hn], 1)
            M_drug = M_drug**24
            M_no_drug = M_no_drug**24
            strat = best_strats[hn][skn]
            rho = get_strat_growthrate(strat, M_drug, M_no_drug, interval_len = interval_len)
            sk_best_cv.append(rho)

        prev_best = best_strats[hn][1]
        cv_strats, cv_mdds = [], []
        for skn, sk in enumerate(np.arange(3.0,12.0,0.25)):
            sig = 6.77 * 10**(-sk)
            M_drug, M_no_drug = get_one_hour_matrices(sig, xb, hedges[hn], 1)
            M_drug = M_drug**24
            M_no_drug = M_no_drug**24
            rho_strat = get_strat_growthrate(prev_best, M_drug, M_no_drug, interval_len = interval_len)
            rho_mdd = get_strat_growthrate(mdd, M_drug, M_no_drug, interval_len = interval_len)
            cv_strats.append(rho_strat)
            cv_mdds.append(rho_mdd)
    
        np.save('./VaryingSigma/Curves/'+hedge_names[hn]+'_'+str(gs_num)+'_'+str(dose), [sk_best_cv, cv_strats, cv_mdds])


#######################################################
# Plots the curves in Figs 5.19 and 5.20
#######################################################
def plot_sig_curves(gs_num, dose):
    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
    'figure.subplot.wspace' : 0.7,
    'figure.subplot.hspace' : 0.5,
    'figure.subplot.bottom' : 0.25,
    'figure.subplot.top' : 0.85
    }
    mpl.rcParams.update(params)

    cols = [sns.xkcd_rgb['sage green'], sns.xkcd_rgb['pale purple'], sns.xkcd_rgb['tangerine'], sns.xkcd_rgb['red'], sns.xkcd_rgb['blue'], sns.xkcd_rgb['green'], sns.xkcd_rgb['yellow'], sns.xkcd_rgb['purple'], sns.xkcd_rgb['claret']]
    plt.figure(figsize = (15,3))
    for hn in range(len(hedges)):
        plt.subplot(1, len(hedges), hn+1)
        data = np.load('./VaryingSigma/Curves/'+hedge_names[hn]+'_'+str(gs_num)+'_'+str(dose)+'.npy')
        sk_best_cv = []
        cv_strats = data[1]
        cv_mdds =  data[2]

        sks = np.arange(3.0,12.0,0.25)
        plt.plot(map(lambda x :6.77 * 10**(-x), sks), cv_strats, ls='--', c = sns.xkcd_rgb['denim blue'], label="MDD")
        plt.plot(map(lambda x :6.77 * 10**(-x), sks), cv_mdds, c = sns.xkcd_rgb['pale red'], label="Opt")
        plt.plot(map(lambda x :6.77 * 10**(-x), sks), [0.0 for i in sks], c = sns.xkcd_rgb['green'], ls = ':')

        plt.title(hedge_names[hn] + " Switch")
        plt.xlabel("Decay Rate (Sigma)")
        plt.xscale('log')
        if hn == 0:
            plt.ylabel("Growth Rate (per week)")
        if hn == len(hedges)-1:
            plt.legend()
        plt.ylim(-0.2, 2.0)

    plt.show()
