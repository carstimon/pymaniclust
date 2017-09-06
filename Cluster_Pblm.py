import pymanopt as mo
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent

from Y_mani import Y_mani
from random_creators import dist_mat

from sklearn.cluster import KMeans

import numpy as np
import numpy.linalg as la

import itertools
import collections

MAXITER = 10000
MINGRADNORM = .0005

class Cluster_Pblm:
    """
    A class which holds all of the data needed for running the clustering problems
    Also has methods for the common functions we use,
    and has our algorithms.

    This is useful because we want to use P and the precomputed vector of the norms of points,
    instead of the distance matrix,
    for computing various things.
    """

    
    def __init__(self, P, k, testing=True):
        """
        P is a (d x n) matrix holding our points.
        k is the number of clusters expected
        If testing is true then we do extra computations which are useful but not computationally necessary.
        """
        self.P = P #Matrix of points
        self.nu = np.sum(P**2, axis=0)[:,None] #Matrix of pts' norms.
        self.d = P.shape[0] #Dimension the points lie in
        self.n = P.shape[1] #number of points
        self.k = k #number of clusters
        self.M = Y_mani(self.n, self.k) #"Y" manifold corresponding to our problem.
        #NB it may be quicker to use "sums" and such strictly instead of the ones matrix
        self.one = np.ones((self.n,1)) #Matrix of all ones.
        #The solver we use for gradient descent.
        #NB there may be some better way to set the settings,
        #Perhaps dynamically depending on the data.
        self.solver = SteepestDescent(maxiter=MAXITER, logverbosity = 1, mingradnorm = MINGRADNORM)
        
        if testing: #Computationally expensive, and not needed in final code.
            self.D = dist_mat(P)**2 #Distance squared matrix
            #Slow way which is replaced by the way below.
            #self.Dsize = la.norm(self.D)

        self.Dsize = np.sqrt(2*self.n*(self.nu**2).sum() + 2*(self.nu.sum())**2
                              + 4*la.norm(P.dot(P.T))**2 - 8*(P.T.dot(P.dot(self.nu)).sum()))
        

    def tr(self, Y):
        """
        Returns tr(DYY^T)
        This code uses the computation which is linear in n.
        """
        nu, P, one = self.nu, self.P, self.one
        term1 = 2*((one.T.dot(Y)).dot(Y.T.dot(nu)))[0,0]
        term2 = -2*np.sum(P.dot(Y)**2)
        return (term1 + term2)
        
    def gr_tr(self, Y):
        """
        Returns the (euclidean) gradient of tr
        This code uses the computation which is linear in n.
        """
        nu, P, one = self.nu, self.P, self.one
        return (2*(one.dot(nu.T.dot(Y))
                   + nu.dot(one.T.dot(Y)))
                - 4*P.T.dot(P.dot(Y)))

    def gr_tr_projected(self, Y):
        """
        Returns the M gradient of tr
        """
        W = self.gr_tr(Y)
        return self.M.proj(Y, W)
        
    def neg(self, Y):
        """
        Returns the norm of the negative part of Y.
        """
        negpt = Y*(Y<0)
        return (negpt**2).sum()

    def gr_neg(self, Y):
        """
        Returns the (euclidean) gradient of the negative part of Y
        """
        return 2*Y*(Y<0)

    def fn_weighted(self, a, b):
        """
        Returns a function which computes
        a*neg(Y) + b*tr(Y)
        """
        return lambda Y: a*self.tr(Y) + b*self.neg(Y)

    def gr_weighted(self, a, b):
        """
        Returns a function which computes
        a*gr_neg(Y) + b*gr_tr(Y)
        """
        return lambda Y: a*self.gr_tr(Y) + b*self.gr_neg(Y)


    def run_minimization(self, a, b, Y0 = None, testing=True):
        """
        Optimizes the problem a*tr(Y) + b*neg(Y)
        If Y0 is given then this runs gradient descent starting from Y0,
        otherwise it starts from a random point.
        Returns the Y value, together with the log information from the solver.
        
        if testing=True then it also prints the number of iterations needed for convergence,
        which is nice for watching the algorithm run and understanding its difficulties.
        """
        cst = self.fn_weighted(a,b)
        grad = self.gr_weighted(a,b)
        pblm = mo.Problem(manifold = self.M,
                          cost = self.fn_weighted(a,b),
                          egrad = self.gr_weighted(a,b),
                          verbosity = 0)
        Y,log = self.solver.solve(pblm, x=Y0)
        print("Number of Iterations: " + str(log['final_values']['iterations']))
        return Y,log

    def run_lloyd(self):
        """
        Run Lloyd's algorithm from sklearn.
        Return the Y matrix corresponding to the clustering given.
        """
        #Note this method expects the transpose of our T matrix.
        clustering = KMeans(n_clusters = self.k).fit(self.P.T) 
        clusters = clustering.labels_
        Y = np.zeros((self.n, self.k))
        cts = collections.Counter(clusters)
        for (pt, cluster) in zip(range(self.n), clusters):
            Y[pt, cluster] = 1/np.sqrt(cts[cluster])
        return Y
            
    def do_path(self, As, Bs, smart_start = True, save=False):
        """
        As and Bs are lists of a and b coefficients.
        Runs the minimization for the (a,b) pairs succesively, using the previous answer as the initial guess.
        If smart_start = True,
        Then the first minimization is done iwth (a,b) = (1,0) and the negative part is minimized as well.
        If save = True,
        Then the list of minimizing Y's is saved and returned
        """
        if smart_start:
            #We first run the minimization of tr alone.
            Y0,_ = self.run_minimization(1, 0)
            #Next we minimize neg(Y) over all Y with the same X = YY^T
            Y = self.minneg_in_SOk(Y0)
            record = [Y]
        else:
            Y = None
        for (a,b) in zip(As,Bs):
            print("Current (a,b): " + str((a,b)))
            Y_prev = Y
            Y,_ = self.run_minimization(a, b, Y0 = Y)
            print("Clustering Change: " + str(la.norm(self.M.round_clustering(Y) - self.M.round_clustering(Y_prev))))
            if save:
                record.append(Y)
        Y_last,_ = self.run_minimization(0, 1, Y0 = Y)
        record.append(Y_last)
        return (Y, record)

    def minneg_in_SOk(self, Y0):
        """
        Minimizes the negative part of $Y_0 Q$ over $Q \in SO(k)$.
        """
        def cost(Q):
            Y = Y0.dot(Q)
            return self.neg(Y)

        def cost_grad(Q):
            Y = Y0.dot(Q)
            return Y0.transpose().dot(Y*(Y<0))
            
        k = Y0.shape[1]
        SOk = Stiefel(k,k)
        pblm =   mo.Problem(manifold = SOk,
                            cost = cost,
                            egrad = cost_grad,
                            verbosity=0)

        Q,log = self.solver.solve(pblm)
        return Y0.dot(Q)



        

