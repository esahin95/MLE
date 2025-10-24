# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:33:25 2025

@author: sahin
"""

import numpy as np
import matplotlib.pyplot as plt
from MLTools import *

def run01():
    ''' 
    Run code for exercise 01. Predict falling time of a point mass with 
    linear regression model. Data generated with ODE integrator
    '''
    # seeded random number generator
    rng = np.random.default_rng(0)
    n = 20
    
    # random initial conditions
    v = rng.uniform(-0.5, 0.5, (n,1))
    h = rng.uniform( 0.1, 2.0, (n,1))
    
    # environment
    g = 9.81
    
    # integrate ODE
    ode = FreeFall(g)
    t = np.zeros((n,1))
    for i in range(n):
        t[i] = ode.run(np.array([h[i], v[i]]), 1e-3)
    
    # nondimensionalized input
    X = v / np.sqrt(g * h)
    
    # nondimensionalized output
    y = t * np.sqrt(g / h)
    
    # build model
    model = LinearRegression(nFeatures=1)
    model.fit(X, y)
    print(model)
    
    # test on moon
    g, h, v = 1.62, 1.5, 0.1
    x = v / np.sqrt(h * g)
    tTrue = np.sqrt(h / g) * (x + np.sqrt(x**2 + 2.0))
    tPred = np.sqrt(h / g) * model(x.reshape(1,1))
    print(tTrue, tPred)
   

def run02():
    ''' 
    Run code for exercise 02. Metropolis algorithm to sample the posterior 
    distribution for coin flips.
    '''    
    rng = np.random.default_rng(0)

    # imports
    from scipy.stats import beta
    
    # data
    p = 0.3 
    n = 20
    X = rng.random(n) < p
    
    # prior probability density
    def prior(x, a=2, b=2):
        return x**(a-2) * (1-x)**(b-2)
    
    # likelihood
    h = np.sum(X)
    def like(x):
        return x**h * (1-x)**(n-h)
    
    # stationary distribution
    def f(x):
        return like(x) * prior(x)
    
    # build Markov chain
    mcmc = MCMC()
    D = mcmc.run(0.5, f, 10000, 100, 1, 0.1)    
    
    # postprocessing
    fig, axs = plt.subplots(1,2)
    axs[0].hist(D, density=True)
    Z = np.linspace(0,1,1000)
    axs[0].plot(Z, beta.pdf(Z, 2, 2), 'b')
    axs[0].plot(Z, beta.pdf(Z, 2+h, 2+n-h), 'r')
    axs[1].plot(D, range(len(D)))
    plt.show()
    

def run02e():
    ''' 
    Run code for Ising model. Magnetization in thermal Equilibrium
    '''
    
    # imports
    from scipy.ndimage import convolve, generate_binary_structure
    
    # mesh
    N = 50
    
    # initial state
    init_random = np.random.random((N,N))
    lat = np.zeros((N,N))
    lat[init_random>=0.75] = 1 
    lat[init_random<0.75] = -1 
    plt.imshow(lat)
    plt.show()
    
    # total energy
    kern = generate_binary_structure(2,1)
    kern[1][1] = False
    def get_energy(lattice):
        arr = -lattice * convolve(lattice, kern, mode='constant')
        return arr.sum() / 2.0
    
    #def metropolis(lat, times, BJ, energy):
    # params
    times = 200000
    BJ = 0.7
        
    # run metropolis
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    energy = get_energy(lat)
    for t in range(0, times-1):
        # random index
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)
        
        # propose new spin
        si = lat[x,y]
        sf = -1 * si 
        
        # compute energy change
        Ei = 0 
        Ef = 0
        if x > 0:
            Ei += -si * lat[x-1,y]
            Ef += -sf * lat[x-1,y]
        if x < N-1:
            Ei += -si * lat[x+1,y]
            Ef += -sf * lat[x+1,y]
        if y > 0:
            Ei += -si * lat[x,y-1]
            Ef += -sf * lat[x,y-1]
        if y < N-1:
            Ei += -si * lat[x,y+1]
            Ef += -sf * lat[x,y+1]
        dE = Ef-Ei
        
        # change state
        if np.random.random() < min(1, np.exp(-BJ*dE)):
            lat[x,y] = sf 
            energy += dE
        
        net_spins[t] = lat.sum()
        net_energy[t] = energy
        
    # postprocessing
    fig, axs = plt.subplots(1,2)
    axs[0].plot(net_energy)
    axs[1].plot(net_spins / N**2)

    
def run03():
    ''' 
    Run code for exercise 02. Predict part failure for given temperature. 
    Test Perceptron and logistic regression model. Perceptron fails, since 
    data is not linearly seperable
    '''
    test = "Logistic"
    
    # dataset from MLE book
    data = DataCollection()
    data.load("Data/partFailure.npz")
    print(data.X, data.y, sep='\n')
    
    # build model
    match test:
        case "Logistic":
            model = LogisticRegression(1)
            model.fit(data.X, data.y, alp=1.0, epochs=10)
            
        case "Perceptron":
            model = Perceptron(1)
            model.fit(data.X, data.y, alp=0.1, epochs=300)
            
        case _:
            raise ValueError("unknown model type")
    print(model)
    
    # compare prediction
    plt.scatter(data.X, data.y, c='b')
    plt.scatter(data.X, model(data.X), facecolors='none', edgecolors='r')
    plt.show()
    

def run03e():
    '''
    Additional content for exercise 02. Multi-class classification problem
    with multinomial logistic regression
    '''
    # generate synthetic dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10,1))
    y = ((X>0.5) * 2 + (X<-0.5) * 1).astype(np.int32)
    
    # build model
    model = MultinomialLogisticRegression(1, 3)
    model.fit(X, y, alp=1.0, epochs=100)
    
    # post processing
    plt.scatter(X,y)
    plt.scatter(X, model(X), facecolors='none', edgecolors='r')
    plt.show()

def run04():
    ''' 
    Run code for exercise 04. Implement regularization into linear model.
    Parameter identification in ODEs.
    '''
    
    # pendulum
    b, m, l, g = 0.1, 0.25, 2.5, 9.81
    ode = Pendulum(b, m, l, g)
    
    # generate dataset
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(2000,2)) * [np.pi, 10.0*np.sqrt(l/g)]
    Y = np.array([ode.rhs(x, 0.)[1:2] for x in X]) + 0.1 * rng.normal(size=(X.shape[0], 1))
    X = np.hstack((X, np.sin(X), np.cos(X)))
    X = np.power(X[...,np.newaxis],np.arange(1,4)).reshape(X.shape[0],-1)
    
    # build model
    model = RidgeRegression(X.shape[-1])
    model.fit(X[:1000], Y[:1000], lam=1.)
    
    # post process
    YPred = model(X[1000:])
    plt.scatter(Y[1000:],YPred)

def run05():
    ''' 
    Naive Bayes
    '''
    pass
    
def run06():
    ''' 
    Support Vector Regression
    '''
    pass
    
def run07():
    ''' 
    Artificial Neural Networks
    '''
    
    # ground truth
    rng = np.random.default_rng(0)
    base = MLP([2,2,2], seed=42)
    X = rng.random((250,2))
    Y = base(X)
    
    # structured mesh
    U,V = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
    P = np.hstack((U.reshape(-1,1), V.reshape(-1,1)))
    W = base(P)
    
    # fit neural network
    model = MLP([2,2,2], seed=659)
    L = model.fit(X, Y, lr=0.01, bs=X.shape[0], epochs=100000)
    YP = model(X)
    WP = model(P)
    
    # plot 
    fig, axs = plt.subplots(1,2,figsize=(10,5), subplot_kw={'projection':'3d'})
    for i in range(Y.shape[-1]):
        # training data
        axs[i].scatter(X[:,0:1], X[:,1:2], Y[:,i:i+1], c='b', s=1)
        axs[i].scatter(X[:,0:1], X[:,1:2], YP[:,i:i+1], c='r', s=1)
        
        # structured surface plot
        axs[i].plot_surface(U, V, W[:,i].reshape(U.shape), color='b', alpha=0.2)
        axs[i].plot_surface(U, V, WP[:,i].reshape(U.shape), color='r', alpha=0.2)
    plt.show()
    
    # training loss
    fig, ax = plt.subplots(1,1)
    ax.semilogy(L[0:])
    plt.show()
    
def run08():
    ''' 
    Use pytorch for image classification
    '''
    pass

def run09a():
    ''' 
    Clustering (k-Means)
    '''
    # load data
    X = np.loadtxt('./Data/clusters.dat')
    
    # build model
    model = KMeans()
    k = 3
    model.fit(X,k)
    
    # post process
    centers = model(X)
    for i in range(k):
        I = (centers == i).reshape(-1)
        plt.scatter(X[I,0], X[I,1], c=f'C{i}', alpha=0.5)
        plt.scatter(model._centroids[i][0], model._centroids[i][1], c='k', alpha=1.0)
    plt.show()
    
def run09b():
    ''' 
    factorization (PCA)
    '''
    # load data
    X = np.loadtxt('./Data/pca.dat', delimiter=',')
    
    # build model
    model = PCA()
    model.fit(X)
    
    # post process
    fig, axs = plt.subplots(16,16,figsize=(10,10),subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw={'hspace': 0, 'wspace': 0})
    for i, ax in enumerate(axs.flat):
        ax.imshow(model[i][1].reshape(16,16), cmap = 'gray')
    plt.tight_layout()
    plt.show()
    
def run10():
    ''' 
    Independent Component Analysis
    '''
    # load data
    X = np.loadtxt('./Data/ica.dat', delimiter=',')
    
    # build model
    model = ICA()
    model.fit(X, lr=5e-4, bs=100, epochs=10)
    
    # post process    
    fig, axs = plt.subplots(16,16,figsize=(10,10),subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw={'hspace': 0, 'wspace': 0})
    for i, ax in enumerate(axs.flat):
        ax.imshow(model[i].reshape(16,16), cmap = 'gray')
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    #run01()
    #run02()
    #run02e()
    #run03()
    run10()