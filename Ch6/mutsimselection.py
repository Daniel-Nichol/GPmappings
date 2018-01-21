##############################################
# D. Nichol 24/04/2017
# Simulation of mutational accumulation
# Following 1/f and M. Williams
##############################################
import sys
from copy import copy
import numpy as np

##############################################
# Primes
##############################################
def primeGenerator():
    with open('./primes.txt', 'r') as pfile:
        line = pfile.readline()
        while line:
            line = pfile.readline()
            prime = int(line.split(',')[1][1:])
            yield prime


#Control flow global state
primes = primeGenerator()
total_muts = 0
driver_found = False

# Change this parameter to vary early/mid/late
# drivers.
driver_count = 10000
drugged = False

#Simulation Parameters
rate = np.log(150)*1./(7*24.)
br = 1./24
dr = br - rate
drugEffect = 10
mu = 1./10.
drugSize = 2.5 * 10**3
finSize = 1.02 * drugSize


##############################################
# Simulation
##############################################
def update(pop1, pop2, br1, br2, dr1, dr2, mu):
    #Determine the event and time to event
    r1,r2 = np.random.random(), np.random.random()
    a = [br1*len(pop1), dr1*len(pop1), br2*len(pop2), dr2*len(pop2)]
    a_0  = sum(a)
    tau = (1./a_0)*np.log((1./r1))
    r2 = a_0*r2

    if a_0 == 0:
        return 1.0, [], []
    
    if r2 <= a[0]:
        bi = np.random.randint(0,len(pop1))
        rep = pop1[bi]
        pop1 = pop1[:bi] + pop1[bi+1:]
        o1, o2, is_driver = reproduce(rep, mu)
        
        #Note: We only allow pop1 -> pop2 driver muts (no back mutations) 
        if is_driver:
            pop2 = extend(pop2, o1, o2)
        else:
            pop1 = extend(pop1, o1, o2)
    
    elif r2 <= sum(a[:2]):
        di = np.random.randint(0, len(pop1))
        pop1 = pop1[:di] + pop1[di+1:]
    
    elif r2 <= sum(a[:3]):
        bi = np.random.randint(0,len(pop2))
        rep = pop2[bi]
        pop2 = pop2[:bi] + pop2[bi+1:]
        o1, o2, _ = reproduce(rep, mu)
        pop2 = extend(pop2, o1, o2)
    
    elif r2 <= sum(a):
        di = np.random.randint(0, len(pop2))
        pop2 = pop2[:di] + pop2[di+1:]
    
    return tau, pop1, pop2

def reproduce(rep, mu):
    nmuts = np.random.poisson(1./mu)
    o1 = rep
    global total_muts
    global driver_found
    for _ in range(nmuts):
        o1 = o1*primes.next()
        total_muts += 1
    
    driver = False
    if (not driver_found) and total_muts > driver_count:
        driver_found = True
        driver = True   
    o2 = o1
    return o1, o2, driver

def extend(pop, o1, o2):
    pop.append(o1)
    pop.append(o2)
    return pop

def sim(finSize, drugSize, br1, br2, dr1, dr2, drugEffect, mu):
    ts, history = [], []
    pop1, pop2 = [2] , []
    sizes = ([], [])
    t = 0
    global drugged
    while len(pop1) + len(pop2) < finSize and len(pop1) + len(pop2) > 0:
        print >> sys.stderr, t, len(pop1), len(pop2)
        ts.append(t)
        sizes[0].append(len(pop1))
        sizes[1].append(len(pop2))
        tau, pop1, pop2 = update(pop1, pop2, br1, br2, dr1, dr2, mu)
        t += tau
        if not drugged and len(pop1) + len(pop2) > drugSize:
            history.append([copy(pop1), copy(pop2)])
            dr1 = drugEffect*dr1
            drugged = True
    
    history.append([copy(pop1), copy(pop2)])
    
    return ts, history, sizes

##############################################
# Post processing 
##############################################
def countmuts(pop, maxp):
    counts = []
    primes2 = primeGenerator()
    pr = primes2.next()
    while pr < maxp:
        print >> sys.stderr, pr, maxp
        num = sum([((p % pr) == 0) for p in pop])
        counts.append(float(num) / len(pop))
        pr = primes2.next()

    return counts

def extractcounts(hist):
    maxp = primes.next()
    pop1 = hist[0][0] + hist[0][1]
    pop2 = hist[1][0] + hist[1][1]
    cnts1 = countmuts(pop1, maxp)
    cnts2 = countmuts(pop2, maxp)
    return cnts1, cnts2

if __name__ == '__main__':
    run_num = sys.argv[1]
    ts, hist, sizes = sim(finSize, drugSize, br, br, dr, dr, drugEffect, mu)
    if sizes[0][-1] + sizes[1][-1] > 5:
        np.save('./GeneticSimData/curve_'+str(run_num), [ts, sizes])
        cs1, cs2 = extractcounts(hist)
        np.save('./GeneticSimData/predrug_'+str(run_num)+'.npy', cs1)
        np.save('./GeneticSimData/progression_'+str(run_num)+'.npy', cs2)
