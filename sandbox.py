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


if __name__ == "__main__":
    #run01()
    #run02()
    run02e()
    #run03()