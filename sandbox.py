# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 09:33:25 2025

@author: sahin
"""

import numpy as np
import scipy as sp
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
   
#@timeit
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
    m = 5
    X = rng.random(m) < p
    
    # prior probability density
    a, b = 2, 2
    def prior(x):
        if x >= 1.0 or x <= 0.0:
            return 0.0 * x
        else:
            return x**(a-1) * (1-x)**(b-1)
    
    # likelihood
    h = np.sum(X)
    def like(x):
        return x**h * (1-x)**(m-h)
    
    # stationary distribution
    def f(x):
        return like(x) * prior(x)
    
    # hyper parameters
    sigma = 1.0e-1
    lag = 1 
    burnin = 100
    nSample = 20000
    
    # build Markov chain
    mc = MCMC(0.5, sigma)
    
    # burnin period
    for _ in range(burnin):
        mc.step(f)
    
    # collect samples
    D = np.zeros(nSample)
    nAccept = 0
    for i in range(nSample):
        for j in range(lag):
            nAccept += mc.step(f)
        D[i] = mc.state
    print(f'Acceptance probability: {nAccept/(nSample*lag):.4f}') 
    
    # postprocessing
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    axs[0].hist(D, bins=50, density=True)
    Z = np.linspace(0,1,1000)
    axs[0].plot(Z, beta.pdf(Z, 2, 2), 'b')
    axs[0].plot(Z, beta.pdf(Z, a+h, b+m-h), 'r')
    axs[1].plot(D, range(len(D)))
    #plt.savefig('Results/coins.pdf', bbox_inches='tight')
    plt.show()
    
#@timeit
def run02e():
    ''' 
    Run code for Ising model. Magnetization in thermal Equilibrium
    '''    
    # mesh
    N = 50
    
    # initial state
    rng = np.random.default_rng(0)
    L = np.ones((N,N))
    L[rng.random((N,N)) < 0.75] = -1 
    plt.imshow(L)
    plt.show()
    
    # total energy
    def energy(L):
        return np.sum(- L * (np.roll(L,1,axis=0) + np.roll(L,1,axis=1)))
    
    # params
    times = 300000
    nCases = 20
    MM = np.zeros(nCases)
    BJs = np.linspace(0.1,1.0,nCases)
    for case, BJ in enumerate(BJs):
        # reset
        L = np.ones((N,N))
        L[rng.random((N,N)) < 0.75] = -1
            
        # run metropolis
        M = np.repeat(L.sum(), times)
        E = np.repeat(energy(L), times)
        for t in range(1, times):
            # proposal step
            x = rng.integers(N)
            y = rng.integers(N)
            
            # compute energy change
            Ei = -L[x,y] * (L[(x+1) % N, y] + L[(x-1) % N, y] + L[x, (y+1) % N] + L[x, (y-1) % N])
            dE = -2.0 * Ei
            
            # acceptance step
            accept = np.random.random() < min(1, np.exp(-BJ*dE))
            if accept:
                L[x,y] *= -1 
            
            # macroscopic observables
            if accept:
                E[t] = E[t-1] + dE 
                M[t] = M[t-1] + L[x,y] * 2
            else:
                E[t] = E[t-1]
                M[t] = M[t-1]
        
        print(np.mean(M[times//2:]) / N**2)
        MM[case] = np.mean(M[times//2:]) / N**2
        
    # postprocessing
    #print(f'Energy: {E[times//2:].mean()}, Magnetization: {M[times//2:].mean()/N**2}')
    #fig, axs = plt.subplots(1,3,figsize=(12,4))
    #axs[0].plot(E)
    #axs[1].plot(M / N**2)
    #axs[2].imshow(L)
    plt.plot(BJs, MM)
    plt.savefig('Results/phase.pdf', bbox_inches='tight')
    plt.show()

    
def run03():
    ''' 
    Run code for exercise 03. Predict part failure for given temperature. 
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
    fig, ax = plt.subplots(1, 1, figsize=(4,2))
    ax.scatter(data.X, data.y, c='b')
    ax.scatter(data.X, model(data.X), facecolors='none', edgecolors='r')
    ax.set(xlabel='$T$', ylabel='Failure')
    plt.savefig('Results/failure.pdf', bbox_inches='tight')
    

def run03e():
    '''
    Additional content for exercise 03. Multi-class classification problem
    with multinomial logistic regression
    '''
    # generate synthetic dataset
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.5,1.5,size=(10,1))
    y = ((X>0.5) * 2 + (X<-0.5) * 1).astype(np.int32)
    
    # build model
    model = MultinomialLogisticRegression(1, 3)
    model.fit(X, y, alp=5.0, epochs=500)
    
    # evaluate model
    C = model.confusion(X,y)
    print(C)
    print(np.sum(np.diagonal(C))/np.sum(C))
    
    # post processing
    plt.scatter(X,y)
    plt.scatter(X, model(X), facecolors='none', edgecolors='r')
    
    # test inputs
    X = rng.uniform(-2,2,size=(1000,1))
    y = ((X>0.5) * 2 + (X<-0.5) * 1).astype(np.int32)
    plt.scatter(X, y, s=1, facecolors='k')
    plt.scatter(X, model(X), s=1, facecolors='none', edgecolors='y')
    plt.show()
    
def run03f():
    ''' 
    Glass identification example
    '''
    # load data
    from ucimlrepo import fetch_ucirepo 
    glass_identification = fetch_ucirepo(id=42) 
      
    # data (as pandas dataframes) 
    X = glass_identification.data.features.to_numpy()
    y = glass_identification.data.targets.to_numpy()
      
    # build model
    model = MultinomialLogisticRegression(X.shape[-1], np.max(y)+1)
    model.fit(X, y, alp=0.001, epochs=1000000)
    C = model.confusion(X,y)
    print(C)
    print(np.sum(np.diagonal(C))/np.sum(C))
    
#@timeit
def ps01():
    ''' 
    Code for assignment 01. Implements the Gauss Newton method
    '''
    # generate data
    rng = np.random.default_rng(0)
    w = np.array([850.0, 0.23])
    X = rng.uniform(0.1, 0.5, size=(10,1))
    y = w[0] * X ** w[1]
    data = DataCollection(X=X, y=y)
    data.save('Data/GaussNewton.npz')
    
    # model
    class F:
        def __init__(self, X, y):
            # model parameters
            self.weights = np.zeros((2,1))
            
            # data
            self._X = X
            self._y = y
            
        def eval(self):            
            # residual
            h = self._X ** self.weights[1]
            f = self.weights[0] * h 
            r = self._y - f
            
            # derivatives
            gradf = np.hstack((h, f*np.log(X)))
            
            # return tuple
            return r, gradf
        
        def __call__(self, X):
            return self.weights[0] * X ** self.weights[1]
    f = F(X, y)
        
    # optimization
    gs = GN()
    gs.fit(f, eps=1e-10)
    print(f.weights)
    
    # post-processing
    plt.scatter(X, y, color='b')
    plt.scatter(X, f(X), color='r')
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
    
def run04a():
    ''' 
    Some extras for exercise 04. Implement the Nealder Mead simplex method and 
    use it for parameter identification in population dynamics.
    '''
    class Base:
        def __init__(self):
            self._nEval = 0
            
    class F(Base):
        def __call__(self, x):
            self._nEval += 1
            return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2
    
    class G(Base):
        def __init__(self, ode, dt, T, X):
            super().__init__()
            self._ode = ode 
            self._dt = dt
            self._T = T 
            self._X = X
            
        def __call__(self, x):
            self._nEval += 1
            X = self._ode.run(x, self._dt, self._T)
            return np.sum((X - self._X)**2)
    
    # test Nealder-Mead
    opt = NealderMead()
    opt(F(), np.array([-1.9,2.0]), eps=1e-10)
    
    # establish ground truth
    r, K, b = 3.0, 15.0, 2.0
    xTrue = np.array([3.0, 15.0, 2.0])
    T = np.linspace(0.5, 5.0, 10)
    X = b / (b/K + (1 - b/K) * np.exp(-r*T)) 
    
    # parameter identification
    x = opt(G(PopDynamics(), 1e-1, T, X), np.array([5.0, 5.0, 5.0]), eps=1e-8)
    
    # test the solution
    ode = PopDynamics()
    T = np.linspace(0.0, 5.0, 100)
    X = ode.run(x, 1e-1, T)
    Y = b / (b/K + (1 - b/K) * np.exp(-r*T)) 
    plt.plot(T, X)
    plt.plot(T, Y, '--')
    plt.show()

def run04b():
    ''' 
    Some extras for exercise 04. Implements the log barrier method to solve optimization
    with (linear) inequality constraints
    '''                    
    # load data
    data = DataCollection()
    #data.load('Data/lasso.npz')
    data.load('Data/polynom.npz')
    n = data.X.shape[-1]
    
    # objective
    class Objective:
        def __init__(self, X, y, lam=1.0):
            # append bias
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            
            # original problem size
            self._nDim = X.shape[-1]
            
            # modified quadratic term
            I = np.eye(self._nDim)
            A = np.block([X, np.zeros_like(X)])
            self._H = A.T @ A
            
            # modified linear term
            c = np.vstack((np.zeros((self._nDim,1)), np.ones((self._nDim,1))))
            self._b = lam * c - A.T @ y
            
            # modified bias
            self._c = 0.5 * (y.T @ y)[0,0]
            
            # linear inequalities
            self._B = np.block([[I, -I], [-I, -I]])
            
        def __call__(self, x):
            return (x.T @ (0.5 * self._H @ x + self._b))[0,0] + self._c
        
        def eval(self, x):
            # gradient          
            gradL = self._H @ x + self._b
            
            # return gradient and Hessian
            return gradL, self._H
            
        def ineq(self, x):
            return self._B @ x, self._B
        
        def __len__(self):
            return self._nDim
    
    # constrained optimization    
    logB = LogBarrier(Objective(data.X, data.y, lam=1.0))
    logB.optimize(epochs=50, maxIter=50)
    
    # unconstrained optimization
    linR = RidgeRegression(nFeatures=n)
    linR.fit(data.X, data.y, lam=1.0)
    
    # post processing
    x = np.arange(n + 1).reshape(-1,1)
    plt.scatter(x, logB.x[:n+1], s=2, color='r', marker='o')
    plt.scatter(x, linR.weights, s=2, color='b', marker='o')
    plt.show()
    
def run04c():
    # load data
    ds = DataCollection()
    ds.load('Data/wLogistic.npz')
    
    # train model    
    model = WeightedLogistic(nFeatures=ds.X.shape[-1])
    model.fit(ds.X, ds.y, tau=0.01)
    
    # test model   
    yp = model(ds.Xt)
    
    # post processing
    fig, axs = plt.subplots(1, 2, figsize=(6,3))
    P = ds.y.flatten() > 0.0
    N = ds.y.flatten() < 0.0
    for i, y in enumerate([ds.yt, yp]):
        axs[i].contourf(
            ds.Xt[:,0].reshape(ds.sz), 
            ds.Xt[:,1].reshape(ds.sz), 
            y.reshape(ds.sz), 
            cmap='coolwarm', 
            alpha=0.5, 
            vmin=-1.0, vmax=1.0
        )
        axs[i].scatter(ds.X[P,0], ds.X[P,1], color='r')
        axs[i].scatter(ds.X[N,0], ds.X[N,1], color='b')
    plt.show()
    
def run04d():
    ''' 
    Extras for exercise 04. Implement polynomial regression
    '''
    # ground truth
    def f(X):
        return 3*X - X**2 + 2*X**3
    
    # generate data
    rng = np.random.default_rng(0)
    m = 10
    X = rng.uniform(-1, 1, size=(m,1))
    y = f(X) + rng.normal(loc=0.0, scale=0.1, size=(m,1))
    
    # feature map
    p = 8
    Phi = np.power(X[...,np.newaxis], np.arange(1,p+1)).reshape(X.shape[0],-1)
    
    #data = DataCollection(X=Phi, y=y)
    #data.save('Data/polynom.npz')
    
    # train model
    model = RidgeRegression(nFeatures=Phi.shape[-1])
    model.fit(Phi, y, lam=0.0)
    
    # test model
    x = np.linspace(-1, 1, 1000)
    yTest = f(x)
    phi = np.power(x[...,np.newaxis], np.arange(1,p+1)).reshape(x.shape[0],-1)
    yPred = model(phi).reshape(x.shape)
    print(np.mean(np.abs(yPred-yTest)))
    
    # post processing
    plt.plot(x, yTest, 'k-')
    plt.scatter(X, y, color='k')
    plt.plot(x, yPred, 'b-')
    plt.show()
    
def run05():
    ''' 
    Run code for exercise 05. Naive Bayes and a mixed version with real features
    for medical diagnosis.
    '''
    # load dataset
    ds = np.loadtxt('Data/diagnosis.csv')
    X = ds[:,1:6].astype(np.int8)
    Y = ds[:,6:7].astype(np.int8)
    
    # training test split
    rng = np.random.default_rng(45)
    idx = rng.permutation(len(X))
    nTest = int(len(X) * 0.2)
    Xtest, Ytest, Xtrain, Ytrain = X[idx[:nTest]], Y[idx[:nTest]], X[idx[nTest:]], Y[idx[nTest:]]
    
    # train model
    model = NaiveBayes()
    model.fit(Xtrain,Ytrain)
    print(model.confusion(Xtest, Ytest))
    
    Xcar = ds[:,0:1]
    Xcartest, Xcartrain = Xcar[idx[:nTest]], Xcar[idx[nTest:]]
    
    model = NaiveBayesMixed()
    model.fit((Xtrain, Xcartrain),Ytrain)
    print(model.confusion((Xtest, Xcartest), Ytest))
    
def run06():
    ''' 
    Support Vector Classification
    '''
    # generate random data
    rng = np.random.default_rng(10)
    X = rng.uniform(-1,1,size=(120,2))
    
    w = np.array([0.2,0.4])
    b = 0.1
    y = np.sign(X @ w + b).reshape(-1,1)
    
    # train model
    model = SVC(0.2)
    model.fit(X, y, C=50, tol=1e-4, maxPasses=5)
    
    # prediction
    Xt,Yt = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
    Pt = np.vstack((Xt.flat, Yt.flat)).T
    
    yPred = np.sign(model(Pt))
    plt.scatter(Pt[yPred.flat>0][:,0], Pt[yPred.flat>0][:,1], color='y')
    plt.scatter(Pt[yPred.flat<0][:,0], Pt[yPred.flat<0][:,1], color='g')
    
    # training data
    plt.scatter(X[y.flat>0][:,0], X[y.flat>0][:,1], color='r')
    plt.scatter(X[y.flat<0][:,0], X[y.flat<0][:,1], color='b')
    
    # solution
    x = np.linspace(-1,1,2)
    plt.plot(x, -(b + w[0]*x)/w[1], 'k')
    plt.show()
    
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
    #run03e()
    run04b()
    #run04d()
    #run06()
    
    #run04c()