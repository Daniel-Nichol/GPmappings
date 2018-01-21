################################################################
# Simulation of treatment strategies for bet-hedging populations.
#
# Here we explore the efficacy of different lengths of holiday 
# for the different molecular switches. 
################################################################
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns

import hedges
import convergence as cnv
import reactionNetwork as rn
################################################################
# Parameters
################################################################
g_max = 60
DC_hedge = hedges.DC_hedge
DC_hedge_x = hedges.get_DC_hedge(1.1,1.0)
DC_hedge_y = hedges.get_DC_hedge(1.0,1.1)
AM_hedge = hedges.get_AM_hedge(60)

w = [2.0,1.01]
g_max = 60
################################################################
# Agent based model, simulates a death-birth process until
# extinction (or a set time limit is reached)
#
# This function is represented in the blue dashed box in
# Figure 4.9
################################################################
def abm_ext(initA, initB, p, Tlim=12000, abm_params=[1.0,0.015,0.8,0.005]):
    fA, fB, dA, dB = abm_params
    numA, numB = initA, initB
    history = [[numA], [numB]]
    # run until extinction (or a time limit is reached)
    t = 0
    while numA + numB > 0 and t<Tlim:
        Adeath = sp.stats.binom.rvs(numA,dA)
        numA = numA - Adeath
        Bdeath = sp.stats.binom.rvs(numB,dB)
        numB = numB - Bdeath
        
        #Calculate Births
        Abirth = sp.stats.binom.rvs(numA, fA)
        Bbirth = sp.stats.binom.rvs(numB, fB)
        numA = numA + int(Abirth*p) + int(Bbirth*p)
        numB = numB + int(Abirth*(1-p)) + int(Bbirth*(1-p))

        history[0].append(numA)
        history[1].append(numB)
        t+=1

    return t

################################################################
# Performes the simulation in Figure 4.9 as follows:
#
# 1. An expected post-holiday genotype is calculated
# 2. The abm is simulated a number of times to generate a 
#    distribution of extinction times.
################################################################
def post_holiday_extinction(t,g0,f,gpmap,trials, abm_params=[1.0,0.015,0.8,0.005]):
    fA, fB, dA, dB = abm_params
    hedge = gpmap(cnv.expectation_of_hedge(g0,f,gpmap,t), 60)
    print hedge
    t_list = []
    if (hedge * (1-dA) * (1+fA)) + (1-hedge)*(1-dB)*(1+fB) >= 1.0:
        return [20000 for i in range(trials)]

    for i in range(trials):
        print "trial: ", i
        ext_t = abm_ext(int(np.floor(hedge*(10**10))),int(np.ceil((1-hedge)*(10**10))), hedge, abm_params)
        t_list.append(ext_t)
    return t_list

################################################################
# Generates all data for the histograms.
################################################################
def generate_histogram_data(f, gpmap, g0, trials, mapname, save_path, abm_params=[1.0,0.015,0.8,0.005]):
    times = [3000,5000,50000,100000]
    gs = [cnv.expectation_of_hedge(g0, f, gpmap, t) for t in times]
    print "generating data for ", mapname
    for g in gs:
        if not os.path.isfile(save_path+mapname+str(g)+'.npy'):
            print 'Generating Histogram for Genotype: '+str(g)
            t_list = post_holiday_extinction(0,g,f,gpmap, trials, abm_params)
            np.save(save_path+mapname+str(g), t_list)
            print 'Histogram data saved!'
        else:
            print 'Data for genotype ', g, ' exists already!'

################################################################
# Builds the histogram grids shown in Figures 4.10-4.12
################################################################
def build_figure(path):
    times = [0,3000, 5000, 50000, 100000]
    names = ['DCy', 'DC', 'DCx', 'AM']
    f = [2.0,1.01]
    g0s = [53,30,7,30]
    maps = [DC_hedge_y, DC_hedge, DC_hedge_x, AM_hedge]
    
    fig = plt.figure(figsize=(12,12))
    fig_num = 1
    
    for t in times:
        for k in range(len(names)):

            #Getting the appropriate data.
            g = cnv.expectation_of_hedge(g0s[k], f , maps[k], t)
            gp = maps[k]
            hedge=gp(g,g_max)
            hist_data = np.load(path+names[k]+str(g)+'.npy')

            if k == 3 and t>=50000:
                ax = plt.subplot(len(times),len(names), fig_num, axisbg='#a5d1df')
            elif t==0:
                ax = plt.subplot(len(times),len(names), fig_num, axisbg='#bdbdbd')
            elif max(hist_data) > 500:
                ax = plt.subplot(len(times),len(names), fig_num, axisbg ='#a5d1df')
            else:
                ax = plt.subplot(len(times),len(names), fig_num, axisbg='#98dd9b')


            fig_num+=1
            bs = 10
            if k == 3 and t>=50000:
                bs = 20
            
            plt.hist(hist_data, bins = bs, cumulative=0, color='white')
            plt.grid(b=True, which='major', color='#f0f0f0', linestyle='-')

            plt.ylim((0, 1000))
            plt.yticks([0,1000.],['0.0', '0.5'])
            plt.ylabel('Frequency')

            if t==0:
                plt.xlim((0,20000))
            elif max(hist_data) > 500:
                plt.xlim((0,10000))
            else:
                plt.xlim((0,60))
                plt.text(0.05, 0.85,'*',fontsize=40, ha='center', va='center', transform=ax.transAxes)
            
            plt.xlabel('Extinction Time (hours)')
            plt.text(0.75, 0.9,'g='+str(g)+', p='+'{:6.5f}'.format(hedge), ha='center', va='center', transform=ax.transAxes)

    plt.subplots_adjust(hspace=0.5)
    plt.show()

