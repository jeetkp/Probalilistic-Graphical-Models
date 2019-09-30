# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:47:24 2019

@author: jeetp
"""


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as py

# np.random.seed(5)



def prob(mu,x):
    return ((0.5)*np.exp((-0.5)*((x-mu)**2)))

var = 0.25


gibbsSamples= np.zeros((1000,2))
mu= np.zeros(2)
m=0;
k=0;
distribution = np.concatenate((np.random.normal(-5,1,50),np.random.normal(5,1,50)))
np.random.shuffle(distribution)
z=np.zeros(100)
sigma = 1;
lam = 10;
#data= np.zeros((1000,2))
while m!=1000:
#    for ii in range(len(distribution)):
    prob0 = prob(mu[0],distribution)
    prob1 = prob(mu[1],distribution)
    check = prob1/(prob0+prob1);
    z= np.random.binomial(1,check)
    n_1= np.count_nonzero(z);
    n_0 = 100 - n_1;
    sum_0 = 0
    sum_1 = 1;
    for i in range(100):
        if z[i]==0:
            sum_0 = sum_0 + distribution[i]
        else: 
            sum_1 = sum_1 + distribution[i]
    sum_0 = sum_0/n_0
    sum_1 = sum_1/n_1
    new_mu_0 =  ((n_0/sigma**2)/((n_0/sigma**2)+(1/lam**2)))* sum_0     
    new_mu_1 =  ((n_1/sigma**2)/((n_1/sigma**2)+(1/lam**2)))* sum_1
    lam_update = (1/((n_1/sigma**2)+(1/lam**2)))
    mu = np.random.multivariate_normal([new_mu_0,new_mu_1],lam_update* np.eye(2))
    if k>=10000:
        if k%10 ==0:
            gibbsSamples[m]= mu
            m+=1;
#            print(" m " + str(m))
    k+=1

print(np.mean(gibbsSamples,axis=0))
py.scatter(gibbsSamples[:,0],gibbsSamples[:,1],color = 'blue')
py.scatter(gibbsSamples[:,1],gibbsSamples[:,0],color = 'red')
py.show()