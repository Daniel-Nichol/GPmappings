#######################################################
# Implements the exhaustive exploration of metronomic
# treatment strategies that comprises the results of
# Section 5.6 - Implications for Therapy
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
#====================================================== 
# Generates Figure 5.12 which highlights the variance
# in growth rate depending on how a dose is prescribed.
#====================================================== 
def all_curves(xb, hedge, dose = 11, cycles = 4):
    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    # 'figure.subplot.bottom' : 0.3,
    'figure.subplot.wspace' : 0.3,
    'figure.subplot.hspace' : 0.5,
    'figure.subplot.bottom' : 0.1,
    'figure.subplot.top' : 0.95
    }
    mpl.rcParams.update(params)

    num_days = 14
    num_intervals = 14
    interval_len = 7
    M_drug, M_nodrug = get_one_hour_matrices(sigma, xb, hedge, 1)
    M_drug = M_drug**24
    M_nodrug = M_nodrug**24

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


    plt.figure(figsize=(8,8))
    ax1 = plt.subplot(1,1,1)
    cvs, rhos = [], []
    for k in range(2**num_days):
        strat = map(int, "{0:b}".format(k).zfill(num_days))
        if sum(strat) == dose:
            Mcycle = np.identity(M_drug.shape[0])
            for j in strat:
                if j == 1:
                    Mcycle =  Mcycle * (M_drug ** interval_len)
                else:
                    Mcycle =  Mcycle * (M_nodrug ** interval_len)

            rho, _ = spectral_rad(Mcycle)
            rho = np.real(rho)
            cycle_f = (1./14.)*np.log(rho) 
            rhos.append(cycle_f)

            pop = deepcopy(init_pop)
            cv = []
            for cycle in range(cycles):
                for interval in range(num_intervals):
                    for day in range(interval_len):
                        cv.append(np.sum(pop))
                        if strat[interval] == 0:
                            pop = M_nodrug * pop
                        else:
                            pop = M_drug * pop
                    
            cvs.append(cv)

    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=min(rhos), vmax = max(rhos))

    # create a ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])

    for ci in range(len(cvs)):
        plt.plot(cvs[ci], c = s_m.to_rgba(rhos[ci]))
    
    cb = plt.colorbar(s_m, ticks=np.arange(min(rhos), max(rhos)+0.0001, (max(rhos) - min(rhos))/8.))
    cb.set_label("One week Growth Rate")

    plt.yscale('log')
    plt.xlim(0,len(cvs[0]))
    plt.ylim(0, 10**14)
    plt.xlabel("Days")
    plt.title("Metronomic Therapy Growth Curves")

    plt.show()

#====================================================== 
# Determined the per-week growth rate for every metronomic
# therapy of length 'num_intervals'.
#
# These values are derived for every combination of 
#       - xb in {30,59}
#       - switch in {DC, DCx, DCy, AM}
#       - memory mechanism in {SG, EM, CM}
#
# 
# This process is extremely slow (>3days on a 2012 MacBook Pro).
#====================================================== 
def generate_all_therapy_data(interval_len = 1, num_intervals = 14):
    print "Warning, takes a LONG time. Uncomment to run..."
    # generate_therapy_data(DCx_hedge, "DCx", 46, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCx_hedge, "DCx", 46, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCx_hedge, "DCx", 46, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DC_hedge, "DC", 59, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DC_hedge, "DC", 59, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DC_hedge, "DC", 59, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCy_hedge, "DCy", 59, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCy_hedge, "DCy", 59, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCy_hedge, "DCy", 59, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(AM_hedge, "AM", 39, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(AM_hedge, "AM", 39, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(AM_hedge, "AM", 39, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)

    #50/50s
    # generate_therapy_data(DCx_hedge, "DCx", 7, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCx_hedge, "DCx", 7, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCx_hedge, "DCx", 7, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DC_hedge, "DC", 30, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DC_hedge, "DC", 30, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DC_hedge, "DC", 30, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCy_hedge, "DCy", 53, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCy_hedge, "DCy", 53, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(DCy_hedge, "DCy", 53, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(AM_hedge, "AM", 30, mem_strat = 0, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(AM_hedge, "AM", 30, mem_strat = 1, interval_len = interval_len, num_intervals = num_intervals)
    # generate_therapy_data(AM_hedge, "AM", 30, mem_strat = 2, interval_len = interval_len, num_intervals = num_intervals)

#######################################################
# Exhaustively evaluates the growth rate for a specified
# space of metronomic therapy.
#
# The results are saved in './Metronomic{interval_len}Days/'
#######################################################
def generate_therapy_data(hedge, name, xb, mem_strat=0, interval_len = 7, num_intervals = 14):
    mem_strat_names = ["EM", "SG", "CM"]
    tot_time = 7*14 #Days
    #The 24hr matrices
    M_drug, M_nodrug = get_one_hour_matrices(sigma, xb, hedge, mem_strat)
    M_drug = M_drug**24
    M_nodrug = M_nodrug**24

    #Storing the strats.
    best_rhos = [np.infty for i in range(num_intervals+1)]
    best_strats = [[] for i in range(num_intervals+1)]
    all_rhos = [[] for i in range(num_intervals+1)]
    mdd_rhos = [0. for i in range(num_intervals+1)]

    for k in range(0, 2**num_intervals):
        print k, "/", 2**num_intervals
        strat = map(int, "{0:b}".format(k).zfill(num_intervals))  

        Mcycle = np.identity(M_drug.shape[0])
        for j in strat:
            if j == 1:
                Mcycle = Mcycle * (M_drug ** interval_len)
            else:
                Mcycle = Mcycle * (M_nodrug ** interval_len)

        rho, _ = spectral_rad(Mcycle)
        rho = np.real(rho)
        cycle_f = (1./14.)*np.log(rho) 

        dose = sum(strat)
        all_rhos[dose].append(cycle_f)

        if "{0:b}".format(k).zfill(num_intervals) == "".join(['1' for i in range(dose)]).ljust(num_intervals, '0'):
            mdd_rhos[dose] = cycle_f

        if cycle_f < best_rhos[dose]:
            best_rhos[dose] = cycle_f
            best_strats[dose] = strat

    folder = './Metronomic'+str(interval_len)+'DayCycles/'
    np.save(folder+'best_rhos_'+mem_strat_names[mem_strat]+'_'+str(num_intervals).zfill(3)+'_'+name+str(xb), best_rhos)
    np.save(folder+'best_strats_'+mem_strat_names[mem_strat]+'_'+str(num_intervals).zfill(3)+'_'+name+str(xb), best_strats)
    np.save(folder+'all_rhos_'+mem_strat_names[mem_strat]+'_'+str(num_intervals).zfill(3)+'_'+name+str(xb), all_rhos)
    np.save(folder+'mdd_rhos_'+mem_strat_names[mem_strat]+'_'+str(num_intervals).zfill(3)+'_'+name+str(xb), mdd_rhos)

#======================================================
# Loads the data generated by the exhaustive search
# defined above.
#======================================================
def load_optimals(mem_strat_name, interval_len, hedgename, xb):
    fileLoc = './Metronomic'+str(interval_len)+'DayCycles/'
    best_strats = np.load(fileLoc + 'best_strats_'+mem_strat_name+'_014_'+hedgename+str(xb)+'.npy')
    best_rhos = np.load(fileLoc+'best_rhos_'+mem_strat_name+'_014_'+hedgename+str(xb)+'.npy')
    return best_strats, best_rhos

def load_mdds(mem_strat_name, interval_len, hedgename, xb):
    fileLoc = './Metronomic'+str(interval_len)+'DayCycles/'
    mdd_rhos = np.load(fileLoc+'mdd_rhos_'+mem_strat_name+'_014_'+hedgename+str(xb)+'.npy')
    return mdd_rhos


#======================================================
# Plots all of the metronomic therapies for each 
# hedge/memorytype.
#======================================================
def make_therapy_plot(interval_len = 7, gnum = 1):
    params = {
       'axes.labelsize': 14,
       'text.fontsize': 14,
       'axes.titlesize': 25,
       'legend.fontsize': 12,
       'xtick.labelsize': 12,
       'ytick.labelsize': 12,
       'text.usetex': False,
       # 'figure.subplot.bottom' : 0.3,
       'figure.subplot.wspace' : 0.4,
       'figure.subplot.hspace' : 0.4
       }
    mpl.rcParams.update(params)

    num_days = 14
    mem_strat_names = ["EM", "SG", "CM"]
    labels = ["Strong Genetic", "Epigenetic Memory", "Constrained Memory"]
    mem_cols = [sns.xkcd_rgb['pale purple'], sns.xkcd_rgb['sage green'], sns.xkcd_rgb['tangerine']]
    mem_cols_darker = [sns.xkcd_rgb['purple'], sns.xkcd_rgb['green'], sns.xkcd_rgb['orange']]
    
    fileLoc = './Metronomic'+str(interval_len)+'DayCycles/'


    alph = 1.0
    marker = "o"
    size = 20

    plt.figure(figsize=(18, 20))
    for hn, name in enumerate(hedge_names):

        xb = gs_sets[gnum][hn]
        best_rhos = []
        for mem_strat in range(3):
            br = np.load(fileLoc + 'best_rhos_'+mem_strat_names[mem_strat]+'_'+str(num_days).zfill(3)+'_'+name+str(xb)+'.npy')
            best_rhos.append(br)

        for mem_strat in range(3):

                ax = plt.subplot(4,3,hn*3 + mem_strat+1)
                all_rhos = np.load(fileLoc + 'all_rhos_'+mem_strat_names[mem_strat]+'_'+str(num_days).zfill(3)+'_'+name+str(xb)+'.npy')
                best_strats = np.load(fileLoc + 'best_strats_'+mem_strat_names[mem_strat]+'_'+str(num_days).zfill(3)+'_'+name+str(xb)+'.npy')
                mdd_rhos = np.load(fileLoc + 'mdd_rhos_'+mem_strat_names[mem_strat]+'_'+str(num_days).zfill(3)+'_'+name+str(xb)+'.npy')
            
                #Other strats
                for alt in range(3):
                    brhos = best_rhos[alt]
                    brhos = map(lambda x : x*(7./interval_len), brhos)
                    plt.plot(brhos, c=mem_cols[alt], zorder=1, ls=styles[alt], lw=2.0)

                #Scatterplot
                for dose in range(len(all_rhos)):
                    rhos = all_rhos[dose]
                    rhos = map(lambda x : x * (7./interval_len), rhos)
                    xs = [dose for i in range(len(rhos))]
                    if dose == 0:
                        plt.scatter(xs, rhos, c = mem_cols[mem_strat], alpha = alph, marker=marker, s=size, label = "Alternative Strategy", zorder=2)
                    else:
                        plt.scatter(xs, rhos, c = mem_cols[mem_strat], alpha = alph, marker=marker, s=size, zorder=2)
                #MDD
                for dose in range(len(all_rhos)):
                    lab = "Maximum Dose Density" if dose==0 else ""
                    plt.scatter([dose], [mdd_rhos[dose]*(7./interval_len)], c = '#525252', marker="H", s=45, label=lab, zorder=2, alpha=0.8)

                plt.xlim(-1,15)
                plt.ylim(-0.5, 2.5)
                plt.xlabel("Total 14 Week Dose\n(drug weeks)", size=22)
                plt.ylabel("Growth Rate\n(per week)", size=22)
                if hn == 0:
                    plt.title(labels[mem_strat])
                ax.text(11.0, 2.3, 'xb = '+str(xb), fontsize=20)

    figname = './FigsTemp/'+str(interval_len)+'_'+str(gnum)+'.png'
    plt.savefig(figname, dpi=500)

#======================================================
# Creates the plot comparing mdd to optimal therapies
# 
# Figs 5.17/5.18
#======================================================
def mdd_opt_comparison(gnum = 0):
    params = {
   'axes.labelsize': 14,
   'text.fontsize': 14,
   'axes.titlesize': 16,
   'legend.fontsize': 12,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': False,
   # 'figure.subplot.bottom' : 0.3,
   'figure.subplot.wspace' : 0.3,
   'figure.subplot.hspace' : 0.5,
    'figure.subplot.bottom' : 0.1,
    'figure.subplot.top' : 0.95
   }
    mpl.rcParams.update(params)

    markers = ['o', '^', 's']
    alphas = [0.9, 0.9, 0.9]
    zorders = [2,2,1]

    titles = ["One Week Intervals", "One Day Intervals"]

    plt.figure(figsize = (7,10))
    for hn, hedgename in enumerate(hedge_names):
        xb = gs_sets[gnum][hn]
        for i, interval_len in enumerate([7,1]):
            ax = plt.subplot(4,2, (2*hn + i + 1))
            if hn == 0:
                plt.title(titles[i])

            ax.text(0.1, 12.1, 'xb = '+str(xb), fontsize=15)
            for mem_num, mem_strat_name in enumerate(mem_strat_names):
                tempName = ["EM", "SG", "CM"]
                _, best_rhos = load_optimals(tempName[mem_num], interval_len, hedgename, xb)
                mdd_rhos = load_mdds(tempName[mem_num], interval_len, hedgename, xb)
                #build the curve
                minDoseRhos = []
                for m, mdd in enumerate(mdd_rhos):
                    minDose = m
                    for dose, opt in enumerate(best_rhos):
                        if  dose < minDose and opt < mdd:
                            minDose = dose
                    minDoseRhos.append(minDose)

                plt.plot(range(15), minDoseRhos, ls="--", lw=0.8, marker = markers[mem_num],
                 markersize=8., c=cols[mem_num], alpha = alphas[mem_num], zorder = zorders[mem_num], label = labels[mem_num])
    
            plt.plot(range(15), range(15), lw=0.9, ls='--', c='k')
            plt.xlabel("MDD Dose")
            plt.ylabel("MEE Dose")
            plt.xlim(-0.5,14.5)
            plt.ylim(-0.5, 14.5)

    plt.legend(bbox_to_anchor=(1.15, -0.30), loc=0, borderaxespad=0., ncol = 3)

    plt.show()