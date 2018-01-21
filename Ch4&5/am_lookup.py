import numpy as np
from copy import deepcopy
################################################################################
# Computing the look up table for the hedges in an AM network
################################################################################
def get_AM_lookup(n,r1=1.,r2=1.,r3=1.,r4=1.,s1=1.,s2=1.):
    # All possible pairs (x,y) with x+y<=n, x>0, y=>0 x>=y.
    S = [(x,y) for x in range(n+1) for y in range(n+1) if (x>0 and y>=0 and x+y<=n and x>=y)]

    S.sort()

    def markov_matrix():
        return np.matrix([[prob(s1,s2) for s2 in S] for s1 in S])

    def prob((x1,y1), (x2,y2)):
        #If you jump to0 far then 0
        if(abs(x1-x2) + abs(y1-y2) > 1):
            return 0.0
        #if you're on the leftmost column
        if y1 == 0:
            return float(x1==x2 and y2==y1)
        #if you're on the leading diagonal
        elif x1 == y1:
            return float(x1==x2 and y1==y2) 
        #b=0    
        elif x1+y1 == n:
            return 0.5 if ((x1==x2 and y2 == (y1-1)) or (y1==y2 and x2==(x1-1))) else 0
        #Every other position has no self loops
        elif(abs(x1-x2) + abs(y1-y2) == 0):
            return 0.0
        #Valid one-steps on the interior of the graph
        else:
            den = 2.*x1*y1 + y1*(n-(x1+y1)) + x1*(n-(x1+y1))
            if(x2 == x1+1):
                return x1*(n-(x1+y1)) / den
            if(x2 == x1-1):
                return x1*y1 / den
            if(y2 == y1+1):
                return y1*(n-(x1+y1)) / den
            if(y2 == y1-1):
                return x1*y1 / den

    def limit(M):
        old_M = deepcopy(M)
        M = M*M
        while not (old_M == M).all():
            old_M = deepcopy(M)
            M = M*M
        return M

    M = markov_matrix()
    N = limit(M)

    def prob_all_x((x,y)):
        if x==0 and y==0:
            return np.nan
        if x+y > n:
            return np.nan
        if y>x:
            return 1-prob_all_x((y,x))
        else:
            mc_index = S.index((x,y))
            total_prob = 0.0
            #probalility of finding a state with y=0
            for i in range(1,n+1):
                target_index = S.index((i,0))
                total_prob+=N[mc_index, target_index]
            for j in range(1, (n/2)+1):
                target_index = S.index((j,j))
                total_prob += 0.5*N[mc_index, target_index]
            return total_prob

    look_up = [[prob_all_x((i,j)) for j in range(n+1)] for i in range(n+1) ]

    return look_up

def save_lookups(M_max):
    for M in range(2, M_max+1):
        print "Computing GP-map for switch ", str(M)
        gp = get_AM_lookup(M)
        np.save('./AMtables/AM'+str(M).zfill(3)+'.npy', gp)

def load_lookups(M_max):
    fullGP = [[],[]]
    for M in range(2, M_max+1):
        gp = np.load('./AMtables/AM'+str(M).zfill(3)+'.npy')
        fullGP.append(gp)
    return fullGP

def get_AM_hedge(xb, yb):
    M_max = 2*(xb+yb)
    fullGP = load_lookups(M_max)
    
    def AM_hedge(x,y):
        gp = fullGP[x+y]
        return gp[x,y]

    return AM_hedge

if __name__ == "__main__":
    # save_lookups(121)
    fgp = load_lookups(121)
    print len(fgp)
    print fgp[-1].shape









