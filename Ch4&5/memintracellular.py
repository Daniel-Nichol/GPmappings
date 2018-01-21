################################################################
# Exploring the dynamics of intracellular decay and 
# epigenetic inhertience.
#
# Produces the Figures in 5.2.1
################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
sns.set(rc={'image.cmap': 'cubehelix'})
import cmocean

import hedges as hds

#######################################################
# The bistable switches.
#######################################################
g_max = 61
DC_hedge = hds.DC_hedge_xy
DCx_hedge = hds.DCx_hedge_xy
DCy_hedge = hds.DCy_hedge_xy
AM_hedge = hds.get_AM_hedge_load(30,30)


#############################################################
# Dynamics of protein decay.
#############################################################
def p(n,t,k,n_0=60):
    p = np.exp(-1*k*n*t) * sp.special.binom(n_0,n) * (1-np.exp(-1*k*t))**(n_0 - n)
    return p

def expectation(t,k,n_0):
    return n_0 * np.exp(-1*k*t)

#Sample a value n, at time t, with rate k, from p.
def sample(t,k,n_0):
    ns = range(0, n_0+1)
    bs = map(lambda x : p(x,t,k,n_0), ns)
    n = np.random.choice(ns,p=bs)
    return n

def getProbAndMols(n_0, t, k, hedge, x_burst, y_burst, phenA = True):
    n = sample(t,k,n_0) #The number of mols remaining.
    if phenA:
        p_A = hedge(n/2. + x_burst, y_burst)
    else:
        p_A = hedge(x_burst, n/2. + y_burst)

    return n,p_A

#############################################################
# The memory hedge. Returns Prob(A | initial conditions), expected.
#############################################################
def probA(t,n_0,k,hedge, x_burst, y_burst):
    prob = 0.0
    for n in range(n_0+1):
        prob+=p(n,t,k,n_0)*hedge(x_burst+n, y_burst)
    return prob

def probDC(t, n_0, k, x_burst, y_burst):
    return probA(t, n_0, k, DC_hedge, x_burst, y_burst)

def probDCx(t, n_0, k, x_burst, y_burst):
    return probA(t, n_0, k, DCx_hedge, x_burst, y_burst)

def probDCy(t, n_0, k, x_burst, y_burst):
    return probA(t, n_0, k, DCy_hedge, x_burst, y_burst)

def probAM(t, n_0, k, x_burst, y_burst):
    return probA(t, n_0, k, AM_hedge, x_burst, y_burst)

#############################################################
# Some useful visualistion
#############################################################
#how the probability changes over time.
def comparing_decay(x_burst = 30, y_burst = 30):
    params = {
    'axes.labelsize': 14,
    'font.size': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False
    }

    mpl.rcParams.update(params)
    #for the tick label formatting.
    import matplotlib.ticker as mtick
    start = 6.77 * 10**(-10)
    stop = 6.77 * 10**(-3)
    interval = (stop - start) / 20
    max_t = 100. * 60 * 60 * 24.
    t_step = 20. #Increase this values for smoother curves

    number_of_lines = len(np.arange(start, stop, interval))
    #Build the colormap.
    cmap = mpl.cm.Blues
    norm = mpl.colors.LogNorm(vmin=start, vmax = 10.**(-2))
    cm_subsection = np.linspace(0.0, 1.0, number_of_lines) 
    colors = [cmap(x) for x in cm_subsection]

    fig = plt.figure()
    hedges = [DC_hedge, DCx_hedge, DCy_hedge, AM_hedge]
    h_names = ["DC Switch", "DCx Switch", "DCy Switch", "AM Switch"]
    
    ylims_l = [0.0, 0.9, 0.050, 0.0] 
    ylims_u = [1.0, 1.0, 0.1, 1.0]
    plt.figure(figsize=(8,8))
    for h in range(len(hedges)):
        ax = plt.subplot(2,2,h+1)
        i = 0
        sig_list = [6.77 * 10**(-k) for k in np.arange(10,3.1,-(10.-3) / 20.)]
        for k in sig_list:
            ts = np.arange(0.0,max_t,max_t/t_step)
            ls = [probA(t, x_burst+y_burst, k, hedges[h], x_burst, y_burst) for t in ts]
            ts = map(lambda t : t/(24. * 60 *60.), ts)
            ax.plot(ts,ls, c= colors[i])
            i+=1

        ax.set_ylim(ylims_l[h], ylims_u[h])
        ax.set_xlim(ts[1], 10**2)
        ax.set_xscale('log')
        ax.set_xlabel('Cell Age (days)')
        ax.set_ylabel('Probability of Phenotype A')
        ax.set_title(h_names[h])

        #Set the number of sig figs on the y axis labels
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.3f'))

    # ax = fig.add_axes([0.85, 0.10, 0.015, 0.8])
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', format='%.2e')
    # cb1.set_label('Decay rate')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.show()

#How the distribution of p changes over time.
def molsprobtimeplot(sigma = 6.77 * 10**(-7)):
    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.subplot.wspace' : 0.7,
    'figure.subplot.hspace' : 0.5,
    'figure.subplot.bottom' : 0.25,
    'figure.subplot.right' : 0.80,
    'figure.subplot.top' : 0.85
    }

    mpl.rcParams.update(params)

    start = 0.0
    stop = 56.01
    interval = 3.5

    #Build the colormap.
    cmap = cmocean.cm.ice
    norm = mpl.colors.Normalize(vmin=start, vmax = stop)

    number_of_lines = len(np.arange(start, stop, interval))
    cm_subsection = np.linspace(0.0, 1.0, number_of_lines) 
    colors = [cmap(x) for x in cm_subsection]

    fig, ax = plt.subplots()
    i=0
    for t in np.arange(start,stop, interval):
        t2 = t * 60 * 60 * 24
        ls = [p(n,t2,sigma) for n in range(61)]
        plt.plot(ls,c=colors[i])
        i+=1

    plt.xlabel("Number of Molecules (n)")
    plt.ylabel("Probability")
    plt.title("Intracellular Molecular Decay")


    ax = fig.add_axes([0.85, 0.25, 0.015, 0.6])
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', ticks=np.arange(start, stop, 7.0))
    cb1.set_label('Cell Age (days)')

    plt.show()

#How the expectation changes over time.
def exp_over_time(n0=60):

    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.subplot.wspace' : 0.7,
    'figure.subplot.hspace' : 0.5,
    'figure.subplot.bottom' : 0.25,
    'figure.subplot.right' : 0.80,
    'figure.subplot.top' : 0.85
    }
    mpl.rcParams.update(params)

    start = 6.77 * 10**(-10)
    stop = 6.77 * 10**(-3)
    interval = (stop - start) / 20
    max_t = 1000 * 60 * 60 * 24.
    number_of_lines = len(np.arange(start, stop, interval))

    #Build the colormap.
    # cmap = mpl.cm.BuGn
    cmap = cmocean.cm.thermal
    norm = mpl.colors.LogNorm(vmin=start, vmax = stop)
    cm_subsection = np.linspace(0.0, 1.0, number_of_lines) 
    colors = [cmap(x) for x in cm_subsection]

    fig, ax = plt.subplots()

    i = 0
    sig_list = [6.77 * 10**(-k) for k in np.arange(10,3, -7./ 20.)]
    for sigma in sig_list:
        es = []
        ts = np.arange(0.0,max_t,60.0)
        for t in ts:
            es.append(expectation(t,sigma,n0))

        ts = map(lambda x : x / (60*60*24), ts)
        plt.plot(ts,es, c = colors[i])
        i+=1

    sigma = 6.77 * 10**(-7)
    es = []
    ts = np.arange(0.0,max_t,60)
    for t in ts:
        es.append(expectation(t,sigma,n0))

    ts = map(lambda x : x / (60*60*24), ts)
    plt.plot(ts,es, c = sns.xkcd_rgb["denim blue"], lw = 3.0, ls='--')

    plt.xlabel("Cell Age (days)")
    plt.ylabel("Expected Number of Internal Molecules")
    plt.title("Dynamics of Intracellular Decay")
    plt.xscale('log')
    plt.xlim((1./(24*60), 1000.))
    # ax = fig.add_axes([0.85, 0.25, 0.015, 0.6])
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', format='%.2e')
    # cb1.set_label('Decay Rate')

    plt.show()

def fixed_point(x_b, y_b, k, avg_tr):
    B_0 = x_b + y_b
    return B_0 * (1. / (1-0.5*np.exp(-1 * k * avg_tr)))

#Expectation over multiple reproduction steps.
def exp_reproduction_plot(k, x_b, y_b, T_a, T_b):

    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False
    }

    mpl.rcParams.update(params)

    num_cycles = 10
    ts_x = [0.0]
    ts_y = [0.0]
    xs = [x_b+y_b]
    ys = [x_b+y_b]

    for cycles in range(num_cycles):
        next_ts = np.arange(0.01,T_a,10.)
        num_x = xs[-1]
        prev_t = ts_x[-1]
        for t in next_ts:
            ts_x.append(prev_t + t)
            xs.append(expectation(t,k, num_x))
        ts_x.append((cycles+1)*T_a)
        xs.append((0.5)*xs[-1]+(x_b+y_b))
    
    num_cycles = int(T_b * 5 / T_b) 
    for cycles in range(num_cycles):
        next_ts = np.arange(0.01,T_b,10.)
        num_y = ys[-1]
        prev_t = ts_y[-1]
        for t in next_ts:
            ts_y.append(prev_t + t)
            ys.append(expectation(t,k,num_y))
        ts_y.append((cycles+1)*T_b)
        ys.append((0.5)*ys[-1]+(x_b+y_b))

    plt.subplot(121)
    plt.plot(ts_x,xs, c=sns.xkcd_rgb["pale red"], label='x molecules')
    fp = fixed_point(x_b, y_b, k, T_a)
    plt.plot((0.0, ts_x[-1]), (fp, fp), 'k--')

    plt.ylim(0,120)
    plt.ylabel("Number of Molecules (One daughter)")
    
    tickwidth = 3
    tickpos = range(0, int(10*T_a), 60*60*24 * tickwidth)
    ticklabs = map(lambda x : str(int(x/(60*60*24))), tickpos)
    plt.xticks(tickpos, ticklabs)
    plt.xlim(0, 10*T_a)
    plt.xlabel("Time (days)")

    plt.title("Lineage of Phenotype A Individuals")

    plt.subplot(122)
    plt.plot(ts_y,ys, sns.xkcd_rgb["denim blue"],label='y molecules')
    fp = fixed_point(x_b, y_b, k, T_b)
    plt.plot((0.0, ts_y[-1]), (fp, fp), 'k--')
    plt.ylabel("Number of Molecules (One daughter)")
    plt.ylim(0,120)

    tickwidth = 14
    tickpos = range(0, int(5*T_b), 60*60*24 * tickwidth)
    ticklabs = map(lambda x : str(int(x/(60*60*24))), tickpos)
    plt.xticks(tickpos, ticklabs)
    plt.xlim(0, 5*T_b)

    plt.xlabel("Time (days)")
    plt.title("Lineage of Phenotype B Individuals")

    plt.show()

def generations_accumulation_plot(k, x_b, T_r):

    params = {
    'axes.labelsize': 14,
    'text.fontsize': 14,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.subplot.wspace' : 0.7,
    'figure.subplot.hspace' : 0.5,
    'figure.subplot.bottom' : 0.25,
    'figure.subplot.right' : 0.80,
    'figure.subplot.top' : 0.85
    }

    mpl.rcParams.update(params)
    dom = range(30) #how many generations of all-x
    def birth_mols(gen):
        B0 = 2*x_b
        a = np.exp(-1 * k * T_r)
        return B0 * ((1.-(0.5*a)**(gen+1))/(1-0.5*a))

    mols = [birth_mols(i) for i in dom]
    plt.plot(dom,mols,marker='o')
    plt.ylabel("Number of Molecules (One daughter, phen A)")
    plt.xlabel("Generations of A")
    plt.show()
