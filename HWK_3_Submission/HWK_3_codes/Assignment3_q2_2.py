# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:00:19 2019

@author: jeetp
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
data=scipy.io.loadmat('SymptomDisease.mat')
W= data['W']
b= data['b']
p= data['p']
s= data['s']

np.random.seed(5)

def sigma(check):
    value_sigma = (1/(1+np.exp(-check)))
    return value_sigma

def joint(d,W,b,s,p):
    check1= sigma(np.dot(W,d)+b.reshape(200))
    hmm = np.prod((p.T**d)*(1-p.T)**(1-d))* \
          np.prod((check1**s.T)*(1-check1)**(1-s.T))
    return hmm

No_samples=8000

d=np.zeros(50)
d_sampled=np.zeros((No_samples,50))
a=[0]*No_samples
prob_check_old = np.zeros(len(d))
# for k in range(0,len(d)):
k=0;
m=0;
while m!= No_samples:
# while len(d_sampled)!=200:
#     d=np.array(d_sampled[k-1])
    Prob_check_current = np.zeros(len(d))
    for i in range(len(d)):
        d[i]=0;
        d_0= joint(d,W,b,s,p)
        d[i]=1
        d_1= joint(d,W,b,s,p)
        prob1 = d_1/(d_0+d_1)
        Prob_check_current[i]=prob1
        d[i]= np.random.binomial(1,prob1)
    if(k>=25000):
        if k%10 == 0 :
            d_sampled[m]=d
            a[m]= np.sum(np.abs(prob_check_old-Prob_check_current))
            prob_check_old=Prob_check_current
            m=m+1
    k=k+1
# plt.plot(a);
# plt.show();
print(k)
d_sampled_tocount = d_sampled.T
for i in range(0,np.size(d_sampled_tocount,0)):
    non0_count = np.count_nonzero(d_sampled_tocount[i])
    print("Prob of disease " + str(i+1) + " is " + str(non0_count/No_samples))


