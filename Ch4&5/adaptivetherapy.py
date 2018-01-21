#######################################################
# Implements the simulation of adaptive therapy that
# generates the results presented in Section 5.7.1
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
from metronomic import *

#######################################################
# Creates Figures 5.20 and 5.21
#
# Note, this requires that the metronomic code has been 
# run (in the same directory) first so that the 
# dose-restricted optimal therapies are recorded.
#######################################################
def plot_adaptive_curves(hedge, hedgename, gnum, mem_strat, interval_len, cycles, hrs_to_change = 7*24):
    params = {
    'axes.titlesize': 16,
    }
    mpl.rcParams.update(params)

    num_intervals = 14
    num_days = num_intervals
    dose_frac = 0.05
    plt.figure(figsize=(15,12))
    dose = 7
    for dn, dose in enumerate([4,7,11]):
        for hn, hedge in enumerate(hedges):
            plt.subplot(3,4,hn + 4*dn + 1)
            hedgename = hedge_names[hn]
            mdd = [1 for i in range(dose)] + [0 for i in range(num_days - dose)]

            xb = gs_sets[gnum][hn]

            M_drug, M_nodrug = get_one_hour_matrices(sigma, xb, hedge, mem_strat)
            M_drug = M_drug**24
            M_nodrug = M_nodrug**24

            #The initial population:    
            yb = 60-xb
            M_max = 2*(xb+yb)+1
            init_pop = [0. for i in range(4*(xb+yb)+2)]
            init_pop[xb+yb] = hedge(xb,yb) * 1.0
            init_pop[xb+yb+M_max] = (1-hedge(xb,yb))*1.0
            init_pop = np.matrix(init_pop).T
            #Find the stationary (no drug) distribution.
            for t in range(7*24):
                init_pop = M_nodrug * init_pop
            init_pop = init_pop / np.sum(init_pop)

            pop = deepcopy(init_pop)
            cvmdd = []
            for cycle in range(cycles):
                for interval in range(num_intervals):
                    for day in range(interval_len):
                        cvmdd.append(np.sum(pop))
                        if mdd[interval] == 0:
                            pop = M_nodrug * pop
                        else:
                            pop = M_drug * pop


            #The dose-restricted optimal.
            best_strats, _ = load_optimals(mem_strat_names[mem_strat], 7, hedgename, xb)
            dose_opt = best_strats[dose]

            while dose_opt[0] == 0:
                frs = dose_opt[0]
                dose_opt = np.concatenate((dose_opt[1:],[frs]))

            cvopt = []
            pop = deepcopy(init_pop)
            for cycle in range(cycles):
                for interval in range(num_intervals):
                    for day in range(interval_len):
                        cvopt.append(np.sum(pop))
                        if dose_opt[interval] == 0:
                            pop = M_nodrug * pop
                        else:
                            pop = M_drug * pop

            cvds = []
            M = M_drug
            pop = deepcopy(init_pop)
            prev_size = np.sum(pop)
            desc_strat = []
            for cycle in range(cycles):
                for interval in range(num_intervals):
                    for day in range(interval_len):
                        cvds.append(np.sum(pop))
                        pop = M * pop
                    if np.sum(pop) > (1.2 * prev_size) and sum(desc_strat[-num_intervals+1:]) < dose:
                        M = M_drug
                        desc_strat.append(1)
                    else:
                        desc_strat.append(0)
                        M = M_nodrug
                    prev_size = np.sum(pop)


            desc_strat = desc_strat[-14:]
            rho_mdd = get_strat_growthrate(mdd, M_drug, M_nodrug, interval_len = interval_len)
            rho_opt = get_strat_growthrate(dose_opt, M_drug, M_nodrug, interval_len = interval_len)
            rho_dss = get_strat_growthrate(desc_strat, M_drug, M_nodrug, interval_len = interval_len)

            if dn == 0:
                plt.title(hedge_names[hn])

            plt.plot(range(len(cvds)), cvds, label='Dose Skipping')
            plt.xlabel("Days")
            plt.ylabel("Fold Growth")
            plt.plot(range(len(cvmdd)), cvmdd, label='MDD')
            plt.plot(range(len(cvopt)), cvopt, label='Opt')
            plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.15, -0.30), loc=0, borderaxespad=0., ncol = 3)
    plt.show()
