# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 23:33:05 2019

@author: jeetp
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as py

#np.random.seed(5)


def distri(x, mu1, mu2):
    val1 = np.prod(((0.5) * np.exp((-0.5) * (x - mu1) ** 2)) + ((0.5) * np.exp((-0.5) * (x - mu2) ** 2)))
    val2 =  np.prod(((np.exp((-0.5) * (mu1 ** 2)/100)) * (np.exp((-0.5) * (mu2 ** 2)/100))))
    return val1*val2

distribution1 = 0.5 * (np.random.normal(-5, 1, 100)) + 0.5 * (np.random.normal(5, 1, 100))
varsq =0.25
var =0.5
mu = np.zeros(2)
m=0;
k=0;
j=0;
distribution = np.concatenate((np.random.normal(-5,1,50),np.random.normal(5,1,50)))
np.random.shuffle(distribution)
data= np.zeros((1000,2))
while m!=1000:
    j=j+1
    Q = np.random.multivariate_normal(mu, varsq * np.eye(2))
    uniform1 = np.random.uniform(0, 1)
    check1 = distri(distribution, Q[0], Q[1]) * np.prod((norm.pdf(Q[0], mu[0], var) * norm.pdf(Q[1], mu[1], var)))
    check2 = distri(distribution, mu[0], mu[1]) * np.prod((norm.pdf(mu[0], Q[0], var) * norm.pdf(mu[1], Q[1],var)))
#    check1 = distri(distribution, Q[0], Q[1]) * np.prod((norm(mu[0],var).pdf(Q[0]) * norm(mu[1],var).pdf(Q[1])))
#    check2 = distri(distribution, mu[0], mu[1]) * np.prod((norm(Q[0],var).pdf(mu[0]) * norm(Q[1],var).pdf(mu[1])))
    term = check1 / check2;
#    print(term)
    val = min(1, term)
    if uniform1 < val:
        k=k+1
        mu=Q;
        if k%1000 ==0 :
            print(k)
        if k>=100:
            print(Q)
            data[m] = Q
            m=m+1
            print(m)
#            mu=Q;
#print(" Acceptance Rate " , str(float(m/k)))
#print(np.mean(data,axis=0))
#py.scatter(data[:,0],data[:,1],color='blue')
#py.scatter(data[:,1],data[:,0],color= 'red')
#py.show();

print(" Acceptance Rate " , str(float(m/j)))
xxx= np.minimum(data[:,0],data[:,1])
yyy=np.maximum(data[:,0],data[:,1])
print(np.mean(xxx), np.mean(yyy))
py.scatter(xxx,yyy)
py.show();