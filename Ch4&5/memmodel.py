#######################################################
# Implements the discrete time population dynamics 
# model defined in Section 5.2
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

#######################################################
# The bistable switches.
#######################################################
g_max = 61
DC_hedge = hds.DC_hedge_xy
DCx_hedge = hds.DCx_hedge_xy
DCy_hedge = hds.DCy_hedge_xy
AM_hedge = hds.get_AM_hedge_load(30,30)

#######################################################
# Parameters - Section 5.3
#######################################################
sigma = 6.77 * 10**(-7) #Equation 5.14
# Birth Rates (per second)
b_A = 4.74*10**(-6) 
b_B = 4.74*10**(-7)
# Death Rates (per second)
d_A = 1.39*10**(-6) 
d_B = 1.39*10**(-7)
#Death Rates, drug (per second)
d_A_drug = 5.56*10**(-6) * 2.
d_B_drug = d_B

hedges = [DC_hedge, DCx_hedge, DCy_hedge, AM_hedge]
gs_sets = [[30,7,53, 30],[59,46,59,39]]

#######################################################
# Plotting parameters
#######################################################
cols = [sns.xkcd_rgb['pale purple'], sns.xkcd_rgb['sage green'], sns.xkcd_rgb['tangerine']]
styles = ['--', '-', 'dotted'] 
mem_strat_names = ["SG", "EM", "CM"]
labels = ["Strong Genetic", "Epigenetic Memory", "Constrained Memory"]
hedge_names = ["DC", "DCx", "DCy", "AM"]


#======================================================
# Models
#======================================================
#######################################################
# Memory-free matrix models (as in Figure 5.1)
#######################################################
def get_M(p,q,w1,w2):
    return np.matrix([[w1*p,w2*(1-q)],[w1*(1-p),w2*q]])

def get_strongg_M(p,w1,w2):
    return get_M(p,(1-p), w1, w2)

def get_constrained_M(p, w1, w2):
    return get_M(p, p, w1, w2)

#######################################################
# The matrix model for a population with epigenetic 
# memory. This model is defined in Section 5.2
#
#
# Here, we provide fix_p and fix_q that override the
# memory component to fix a bet-hedging strategy such 
# as the strong-genetic or constrained memory strategies
# shown in Figure 5.1. 
#
# The matrix structure is as follows:
#
#    M_max = 2(xb+yb)
#    mat[i][j] = contribution to 'bin i' from 'bin j'
#    mat[0...M_max) are the As
#    mat[M_max...2M_max)
#
#######################################################
def get_memory_M(sigma, d1, d2, f1, f2, xb, yb, hedge = DC_hedge, fix_p=-1, fix_q=-1):

    M_max = 2*(xb+yb)+1
    mat = [[0. for j in range(2*M_max)] for i in range(2*M_max)]

    for n in range(M_max):
        #Contribution from individuals that do not die or decay
        if n > 0:
            mat[n][n] += (1-(n*sigma))*(1-d1)
            mat[n+M_max][n+M_max] += (1-(n*sigma))*(1-d2)
        #Special case, decay cannot occur at 0-molecules
        else:
            mat[n][n] += (1-d1)
            mat[n+M_max][n+M_max] += (1-d2)
        #Contribution from decaying molecules  (n+1 -> n)
        if n < M_max-1:
            mat[n][n+1] += (1-d1) * sigma * (n+1)
            mat[n+M_max][n+M_max+1] += (1-d2) * sigma * (n+1)

    #Accounting for reproduction.
    #For each parent bin, n_p in [0..M_max)
    #We must consider separate cases for n_p
    for n_p in range(0, M_max):
        #The molecules will be split evenly.
        if n_p % 2 == 0:
            #The P(A | n_p) for offspring of phen A or B parents in bin n_p
            p_a,p_b = hedge(n_p/2 + xb,yb), hedge(xb, n_p/2 + yb)

            #Override this for SG or CM if necessary
            if fix_q > -1 and fix_q > -1:
                p_a = fix_p
                p_b = fix_q

            #Contribution to the A-bins
            mat[(n_p/2)+xb+yb][n_p] += (1-d1)*f1*p_a
            mat[(n_p/2)+xb+yb][n_p + M_max] += (1-d2)*f2*p_b

            #Contribution to the B-bins
            mat[(n_p/2)+xb+yb + M_max][n_p] += (1-d1)*f1*(1-p_a)
            mat[(n_p/2)+xb+yb + M_max][n_p + M_max] += (1-d2)*f2*(1-p_b)

        #The molecules will be split unevenly, note n_p/2 is rounded down.
        if n_p % 2 == 1:

            #The P(A | n_p) for offspring of phen A or B parents in bin n_p: (u)pper and (l)ower
            p_a_l,p_b_l = hedge(n_p/2 + xb,yb), hedge(xb, n_p/2 + yb)
            p_a_u,p_b_u = hedge(n_p/2 + 1 + xb,yb), hedge(xb, n_p/2 + 1 + yb)

            #Override this for SG or CM if necessary
            if fix_p > -1 and fix_q > -1:
                p_a_l, p_a_u = fix_p, fix_p
                p_b_l, p_b_u = fix_q, fix_q

            #Contribution to the A-bins
            mat[(n_p/2)+xb+yb][n_p] += (1-d1)*0.5*f1*p_a_l
            mat[(n_p/2)+xb+yb][n_p + M_max] += (1-d2)*0.5*f2*p_b_l
            mat[(n_p/2)+xb+yb+1][n_p] += (1-d1)*0.5*f1*p_a_u
            mat[(n_p/2)+xb+yb+1][n_p + M_max] += (1-d2)*0.5*f2*p_b_u

            #Contribution to the B-bins
            mat[(n_p/2)+xb+yb + M_max][n_p] += (1-d1)*0.5*f1*(1-p_a_l)
            mat[(n_p/2)+xb+yb + M_max][n_p + M_max] += (1-d2)*0.5*f2*(1-p_b_l)
            mat[(n_p/2)+xb+yb+1 + M_max][n_p] += (1-d1)*0.5*f1*(1-p_a_u)
            mat[(n_p/2)+xb+yb+1 + M_max][n_p + M_max] += (1-d2)*0.5*f2*(1-p_b_u)

    return np.matrix(mat)


#######################################################
# Constructs a matrix model for 1hr of population growth
# using the above method. This is P^(60*60) as in Eqn 5.33
#
# The memory stratgies are denoted by integers according
# to:
#
#   0 - Strong Genetic
#   1 - Epigenetic Memory
#   2 - Contrained Memory
#######################################################
def get_one_hour_matrices(sigma, xb, hedge, mem_strat):
    #The population dynamics matrices:
    if mem_strat == 0:
        p = hedge(xb, 60-xb)
        M_drug = get_memory_M(sigma, d_A_drug, d_B_drug, b_A, b_B, xb, 60-xb, hedge, fix_p = p, fix_q = p)**(60*60)
        M_nodrug = get_memory_M(sigma, d_A, d_B, b_A, b_B, xb, 60-xb, hedge, fix_p = p, fix_q = p)**(60*60)       

    if mem_strat == 1:
        M_drug = get_memory_M(sigma, d_A_drug, d_B_drug, b_A, b_B, xb, 60-xb, hedge)**(60*60)
        M_nodrug = get_memory_M(sigma, d_A, d_B, b_A, b_B, xb, 60-xb, hedge)**(60*60)

    if mem_strat == 2:
        p = hedge(xb, 60-xb)
        M_drug = get_memory_M(sigma, d_A_drug, d_B_drug, b_A, b_B, xb, 60-xb, hedge, fix_p = p, fix_q = 1-p)**(60*60)
        M_nodrug = get_memory_M(sigma, d_A, d_B, b_A, b_B, xb, 60-xb, hedge, fix_p = p, fix_q = 1-p)**(60*60)   

    return M_drug, M_nodrug

#======================================================


#======================================================
# Matrix functions for population dynamics
#======================================================
#######################################################
#Returns the spectral radius (rho) and associated eigenvector
#######################################################
def spectral_rad(P):
    evals, evecs = la.eig(P)[0], la.eig(P)[1]
    spec_rad = max(evals)
    i = list(evals).index(spec_rad)
    perron_v = evecs[:,i]
    perron_v = perron_v / sum(perron_v)
    return spec_rad, perron_v

#######################################################
# Computing the Lypunov Exponent
# Returns rho for the periodic environments
#######################################################
def get_lyapunov_exp(M1, M2, k1, k2):
    C = (M1**k1)*(M2**k2)
    spec_rad,_ = spectral_rad(C)
    rho = 1./(k1+k2) * np.log(spec_rad)
    return np.real(rho)


#######################################################
# Returns the growth rate over a number of 'cycles' for 
# a specified drug schedule 'strat' (Eqn 5.39).
#######################################################
def get_strat_growthrate(strat, M_drug, M_nodrug, interval_len = 7):
        Mcycle = np.identity(M_drug.shape[0])
        for j in strat:
            if j == 1:
                Mcycle = (M_drug**interval_len) * Mcycle 
            else:
                Mcycle =  (M_nodrug**interval_len) * Mcycle

        rho, _ = spectral_rad(Mcycle)
        rho = np.real(rho)
        cycle_f = (1./14.)*np.log(rho) 
        return cycle_f

#======================================================
#Fixed environment plot
#
# Plots fitness (rho) against genotype (xb) for each 
# combination for both environments (drug/drug-free) and
# each of the molecular switches
#
# (Figure 5.7)
#======================================================
def fixed_e_plot():
    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    }

    mpl.rcParams.update(params)

    plt.figure(figsize=(12,8))
    for f_index in range(2):
        epistasisData = [[],[],[],[]]
        d_val = d_A if f_index == 0 else d_A_drug
        for h_index, hedge in enumerate(hedges):
            name = hedge_names[h_index]
            fs, fs_sg, fs_cn = [], [], []
            unconstrained_max = 1.0
            for xb in range(0,61): 
                #Full model
                M = get_memory_M(sigma, d_val, d_B, b_A, b_B, xb, 60-xb, hedge)**(60*60*24*7)

                f, _ = spectral_rad(M)
                f = np.log(np.real(f))
                fs.append(f)

                p = hedge(xb,60-xb)
                #The special case for constrained memory hedging
                if xb != 60 or f_index!=1:
                    M_cn = get_memory_M(sigma, d_val, d_B, b_A, b_B, xb, 60-xb, hedge, p, 1-p)**(60*60*24*7)
                else:
                    print "!"
                    M_cn =  get_memory_M(sigma, d_val, d_B, b_A, b_B, xb, 60-xb, hedge, 0.0, 0.0)**(60*60*24*7)

                M_sg = get_memory_M(sigma, d_val, d_B, b_A, b_B, xb, 60-xb, hedge, p, p)**(60*60*24*7)
                fsg, _ = spectral_rad(M_sg)
                fcn, pv = spectral_rad(M_cn)

                fsg = np.log(np.real(fsg))
                fcn = np.log(np.real(fcn))
                fs_sg.append(fsg)
                fs_cn.append(fcn)

            plt.subplot(2, len(hedges), h_index + len(hedges)*f_index + 1)
            plt.plot(fs, c=cols[1], lw=2.0, label="Epigenetic Memory")
            plt.plot(fs_sg, c=cols[0], lw=2.0, ls='--', label="Strong Genetic")
            plt.plot(fs_cn, c=cols[2], lw=2.0, ls='dotted', label="Constrained Genetic")
            if f_index == 0:
                plt.ylim(0.0, 2.5)
                plt.title(hedge_names[h_index])
            
            plt.xlabel('Genotype (xb)')
            plt.ylabel('Growth Rate (per week)')

            epistasisData[h_index].append(fs_sg)
            epistasisData[h_index].append(fs)
            epistasisData[h_index].append(fs_cn)

        np.save('./fixedFs'+str(f_index), epistasisData)

    plt.legend(bbox_to_anchor = (0.4, -0.7), loc='lower center', ncol=4)
    plt.suptitle("Fixed Environment Growth Rates", size=20)
    plt.show()

#======================================================
# The following two methods generate Figure 5.10
#
# generate_cyclic_plot_data - generates the full results
# make_cycling_plot - plots the data
#======================================================
def generate_cyclic_plot_data(hedge, name):
    cycle_len = 28
    plot_data = []
    for k in  [0, 7, 14, 21, 28]:
        fs, fs_sg, fs_cn = [], [], []
        for xb in range(0,61): 
            print k,xb

            M1 = get_memory_M(sigma, d_A, d_B, b_A, b_B, xb, 60-xb, hedge)**(60*60*24*7)
            M2 = get_memory_M(sigma, d_A_drug, d_B, b_A, b_B, xb, 60-xb, hedge)**(60*60*24*7)

            p = hedge(xb,60-xb)
            M_cn_1 = get_memory_M(sigma, d_A, d_B, b_A, b_B, xb, 60-xb, hedge, p, (1-p))**(60*60*24*7)
            M_cn_2 = get_memory_M(sigma, d_A_drug, d_B, b_A, b_B, xb, 60-xb, hedge, p, (1-p))**(60*60*24*7)
            M_sg_1 = get_memory_M(sigma, d_A, d_B, b_A, b_B, xb, 60-xb, hedge, p, p)**(60*60*24*7)
            M_sg_2 = get_memory_M(sigma, d_A_drug, d_B, b_A, b_B, xb, 60-xb, hedge, p, p)**(60*60*24*7)

            f = get_lyapunov_exp(M1, M2, k, (cycle_len-k))
            fsg = get_lyapunov_exp(M_sg_1, M_sg_2, k, (cycle_len-k))
            fcn = get_lyapunov_exp(M_cn_1, M_cn_2, k, (cycle_len-k))
            
            fs.append(f)
            fs_sg.append(fsg)
            fs_cn.append(fcn)

        plot_data.append([deepcopy(fs), deepcopy(fs_sg), deepcopy(fs_cn)])

    np.save('./fluc_plot'+name+'.npy', plot_data)

def make_cycling_plot():

    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    }
    mpl.rcParams.update(params)

    names = ['DC', 'DCx', 'DCy', 'AM']
    for nm in range(len(names)):
        plot_data = np.load('./fluc_plot'+names[nm]+'.npy')
        for k in range(len(plot_data)):
            plt.subplot(len(plot_data), len(names), k*len(names) + nm + 1)
            if k==0:
                plt.title(names[nm])
            fs, fs_sg, fs_cn = plot_data[k][0], plot_data[k][1], plot_data[k][2]        
            plt.plot(range(0,61), fs, c=cols[1], ls=styles[1], label = 'Epigenetic Memory')
            plt.plot(range(0,61),fs_sg, c=cols[0], ls=styles[0], label = 'Strong Genetic')
            plt.plot(range(0,61), fs_cn, c=cols[2], ls=styles[2], label = 'Constrained Memory')
            plt.ylim(-0.1, 2.5)
            plt.ylabel('Growth Rate')
            plt.xlabel('Genotype (xb)')

    plt.legend(bbox_to_anchor = (0.0, -0.7), loc='lower center', ncol=3)
    plt.show()


#======================================================
# Generates Figure 5.11 for specified molecular switch
# and xb value (yb = 60-xb)
#======================================================
def mdd_fig(hedge, xb):
    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    }
    mpl.rcParams.update(params)

    t_drug = 0
    t_max = 7*52
    fig = plt.figure()
    cmap = cmocean.cm.thermal
    norm = mpl.colors.Normalize(vmin=0, vmax = t_max/7.+1)

    # create a ScalarMappable and initialize a data structure
    s_m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])

    for k in range(3):
        ax = plt.subplot(1,3,k+1)
        M_drug, M_nodrug = get_one_hour_matrices(sigma, xb, hedge, k)
        M_drug = M_drug**24
        M_nodrug = M_nodrug**24
        e_mat = M_nodrug

        for t_stop_drug in range(t_drug, t_max+1, 7*4):
            e_mat = M_nodrug
            pop = np.matrix([1.0 for i in range(M_drug.shape[0])]).T
            pop = pop / sum(pop)
            # pop  = e_mat * pop
            curve = []
            curve = [sum(pop)[0,0]]
            for t in np.arange(0,t_max):
                if t > t_drug:
                    if t>t_stop_drug: 
                        e_mat = M_nodrug
                    else:
                        e_mat = M_drug
                pop = e_mat * pop
                curve.append(sum(pop)[0,0])
            plt.plot(curve, c = s_m.to_rgba(t_stop_drug/7.))
        
        if k == 2:
            cb = plt.colorbar(s_m, ticks=range(0, 53, 4))
            cb.set_label("Total Dose (Drug Weeks)")

        plt.yscale('log')
        plt.xlabel('Days')
        plt.xlim(0,365)
        plt.ylim(10.**(-2), 10.**44)
        plt.ylabel('Fold Increase in Tumour Burden')
        plt.title(labels[k])
        ax.text(0.02, 0.95, 'DC, xb = '+str(xb), fontsize=15, transform = ax.transAxes)

    plt.show()
    plt.suptitle("MDD Therapies Fail", size=16)