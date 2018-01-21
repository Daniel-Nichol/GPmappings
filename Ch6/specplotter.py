##############################################################################
# DN 05/05/2017
# Visulizes the mutational spectra of 
# genetic and BH driven resistance
##############################################################################
import sys
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("white")

##############################################################################
# Load the data
##############################################################################
def load_curves(subdir):
    curve_list = []
    for file in glob.glob(subdir+"/curve_*.npy"):
        ts, cvs = np.load(file)
        curve_list.append((ts, cvs[0], cvs[1]))        
    return curve_list

def load_freqs(subdir):
    pre_freqs, post_freqs = [], [] 
    for file in glob.glob(subdir+"/predrug_*.npy"):
        spec = np.load(file)
        pre_freqs.append(spec)
    for file in glob.glob(subdir+"/progression_*.npy"):    
        spec = np.load(file)
        post_freqs.append(spec)

    pre_freqs = map(lambda f : f/2. , pre_freqs)
    post_freqs = map(lambda f : f/2. , post_freqs)
    return zip(pre_freqs, post_freqs)    

##############################################################################
# Binomial sampling
##############################################################################
def bin_sample(freqs, depth):
    return [np.random.binomial(depth, f) / float(depth)  for f in freqs]


def make_spectra(matched_freqs, depth):
    return (bin_sample(matched_freqs[0], depth), bin_sample(matched_freqs[1], depth))

def add_trunk(freqs, depth, num):
    clons1 = [np.random.binomial(depth, 0.5) for _ in range(num)]
    clons2 = [np.random.binomial(depth, 0.5) for _ in range(num)]
    return (freqs[0] + clons1, freqs[1] + clons2)

##############################################################################
# Extracting 'clonal' reads
##############################################################################
def clonal_num(spectra, thresh=0.35):
    return len(filter(lambda x : x>thresh, spectra))

def make_matched_clonals(matched_spectra, thresh=0.35):
    matched_clonals = [(clonal_num(m[0], thresh), clonal_num(m[1], thresh)) for m in matched_spectra]
    return matched_clonals

def delta_clonals(matched_clonals):
    return [c[1]-c[0] for c in matched_clonals]

##############################################################################
# BH vs Genetic
##############################################################################
def compareGPs(depth=200):
    sg_freqs = load_freqs('./GeneticSimData')
    sg_matched_clonals = make_matched_clonals(map(lambda x : make_spectra(x, depth), sg_freqs))
    sg_deltas = delta_clonals(sg_matched_clonals)

    bh_freqs = load_freqs('./BHSimData99mem')
    bh_matched_clonals = make_matched_clonals(map(lambda x : make_spectra(x, depth), bh_freqs))
    bh_deltas = delta_clonals(bh_matched_clonals)

    df = pd.DataFrame([bh_deltas, sg_deltas]).T

    import scipy
    print scipy.stats.ttest_1samp(sg_deltas, 0.0, axis=0)
    print scipy.stats.ttest_1samp(bh_deltas, 0.0, axis=0)

    fig, ax = plt.subplots(figsize=(8,4))
    plt.subplot(1,1,1)
    ax = sns.violinplot(df, inner='box')
    plt.show()

def save_matched_clonals(subdir):
    matched_freqs = load_freqs(subdir)
    matched_spectra = map(lambda x : make_spectra(x, 200), matched_freqs)
    matched_spectra = map(lambda x : add_trunk(x, 200, 300), matched_spectra)
    matched_clonals = make_matched_clonals(matched_spectra)
    np.save(subdir+'_matchedclonal.npy', matched_clonals)

def load_matched_clonals(subdir):
    return np.load(subdir+'_matchedclonal.npy')


def extract_pre_posts(pp):
    pres, posts = [], []
    for (pre, post) in pp:
        additional_clonal = np.random.binomial(600, 0.5)
        pres.append(pre-300+additional_clonal)
        posts.append(post-300+additional_clonal)
    return pres, posts

##############################################################################
# Plotting
##############################################################################
def plot_pre_post():
    bh_pp = load_matched_clonals('BHSimData99sg')
    early_pp = load_matched_clonals('GeneticSimDataEarly')
    mid_pp =  load_matched_clonals('GeneticSimData')
    late_pp =  load_matched_clonals('GeneticSimDataLate')

    linealpha = 0.3

    fig = plt.figure(figsize=(12,3))
    fig.subplots_adjust(top=0.8)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(right=0.95)
    fig.subplots_adjust(left=0.05)

    plt.subplot(141)
    pres, posts = extract_pre_posts(bh_pp)
    df = pd.DataFrame([pres, posts]).T
    pal = {0 : '#d0d1e6', 1 : '#0570b0'}
    ax = sns.violinplot(data=df, inner='box', palette=pal, linewidth=1.0)
    for pre, post in zip(pres, posts):
        ax.plot([(0,pre), (1,post)], ls='--', lw=0.5, alpha=linealpha, c='k')

    plt.ylim(0, 800)
    plt.xticks([0,1], ['Pre-drug', 'Post-drug'])
    plt.ylabel('Clonal Mutations')
    plt.title('Bet-Hedging')
    
    plt.subplot(142)
    pres, posts = extract_pre_posts(early_pp)
    # pres = map(lambda x : x[0], early_pp)
    # posts = map(lambda x : x[1], early_pp)
    pal = {0 : '#fee6ce', 1 : '#fd8d3c'}
    df = pd.DataFrame([pres, posts]).T
    ax = sns.violinplot(data=df, inner='box', palette=pal, linewidth=1.0)
    for pre, post in itertools.izip(pres, posts):
        ax.plot([(0,pre), (1,post)], ls='--', lw=0.5, alpha=linealpha, c='k')

    plt.ylim(0, 800)
    plt.xticks([0,1], ['Pre-drug', 'Post-drug'])
    plt.title('Resistant Mutation #1000 (Early)')


    plt.subplot(143)
    pres, posts = extract_pre_posts(mid_pp)
    # pres = map(lambda x : x[0], mid_pp)
    # posts = map(lambda x : x[1], mid_pp)
    df = pd.DataFrame([pres, posts]).T
    pal = {0 : '#fee6ce', 1 : '#f16913'}
    ax = sns.violinplot(data=df, inner='box', palette=pal, linewidth=1.0)
    for pre, post in itertools.izip(pres, posts):
        ax.plot([(0,pre), (1,post)], ls='--', lw=0.5, alpha=linealpha, c='k')

    plt.ylim(0, 800)
    plt.xticks([0,1], ['Pre-drug', 'Post-drug']) 
    plt.title('Resistant Mutation #5000 (Mid)')

    plt.subplot(144)
    pres, posts = extract_pre_posts(late_pp)
    # pres = map(lambda x : x[0], late_pp)
    # posts = map(lambda x : x[1], late_pp
    pal = {0 : '#fee6ce', 1 : '#d94801'}
    df = pd.DataFrame([pres, posts]).T
    ax = sns.violinplot(data=df, inner='box', palette=pal, linewidth=1.0)
    for pre, post in itertools.izip(pres, posts):
        ax.plot([(0,pre), (1,post)], ls='--', lw=0.5, alpha=linealpha, c='k')

    plt.ylim(0, 800)
    plt.xticks([0,1], ['Pre-drug', 'Post-drug'])
    plt.title('Resistant Mutation #10000 (Late)')

    plt.suptitle('Matched Pre- and Post-Treatment Clonal Load')

    plt.show()

def plot_delta():
    bh_delt =  delta_clonals(load_matched_clonals('BHSimData99sg'))
    early_delt = delta_clonals(load_matched_clonals('GeneticSimDataEarly'))
    mid_delt =  delta_clonals(load_matched_clonals('GeneticSimData'))
    late_delt =  delta_clonals(load_matched_clonals('GeneticSimDataLate'))

    print np.mean(bh_delt), np.mean(early_delt), np.mean(mid_delt), np.mean(late_delt)
    print np.median(bh_delt), np.median(early_delt), np.median(mid_delt), np.median(late_delt)

    df = pd.DataFrame([bh_delt, early_delt, mid_delt, late_delt]).T
    
    fig = plt.figure(figsize=(12,3))
    fig.subplots_adjust(top=0.8)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(right=0.9)
    fig.subplots_adjust(left=0.1)

    ax = plt.subplot(111)
    pal = {0 : '#0570b0', 1 :'#fd8d3c' , 2:'#f16913' , 3:'#d94801' }
    ax = sns.violinplot(data=df, inner='box', pallete=pal, linewidth=1.0)
    plt.xticks([0,1,2,3], ['Bet Hedging', 'Early Mutant', 'Mid Mutant', 'Late Mutant'])
    plt.ylabel('Change in Clonal Mutations')
    plt.suptitle("Change in Clonal Load Post-Therapy")
    plt.show()


def example_curves():
    fig = plt.figure(figsize = (12,3))
    fig.subplots_adjust(top=0.8)
    fig.subplots_adjust(bottom=0.2)
    fig.subplots_adjust(right=0.9)
    fig.subplots_adjust(left=0.1)
    fig.subplots_adjust(wspace=0.3)

    plt.subplot(121)
    cv_bh = np.load('./BHSimData99sg/curve_50.npy')
    plt.plot(cv_bh[0], cv_bh[1][0], c=sns.xkcd_rgb['pale red'], label='Phenotype A')
    plt.plot(cv_bh[0], cv_bh[1][1], c=sns.xkcd_rgb['denim blue'], label='Phenotype B')
    plt.legend()
    plt.title('Bet-Hedging Growth Curve')
    plt.ylabel('Number of Cells')
    plt.xlabel('Time (hrs)')
    plt.xlim((0,max(cv_bh[0])))


    plt.subplot(122)
    cv_gen = np.load('./GeneticSimData/curve_88.npy')
    plt.plot(cv_gen[0], cv_gen[1][0], c=sns.xkcd_rgb['pale red'], label='Phenotype A')
    plt.plot(cv_gen[0], cv_gen[1][1], c=sns.xkcd_rgb['denim blue'], label='Phenotype B')
    plt.legend()
    plt.title('Genetic Resistance Growth Curve')
    plt.ylabel('Number of Cells')
    plt.xlabel('Time (hrs)')
    plt.xlim((0,max(cv_gen[0])))
    plt.show()

# if __name__ == "__main__":
    # example_curves()
    # plot_pre_post()
    # plot_delta()
    # compareGPs(200)
    # subdir = sys.argv[1]
    # save_matched_clonals(subdir)
    # matched_clonals = load_matched_clonals(subdir)
    # box_plot_clonals(matched_clonals)
    # pair_plot_spectra(matched_spectra[0])
