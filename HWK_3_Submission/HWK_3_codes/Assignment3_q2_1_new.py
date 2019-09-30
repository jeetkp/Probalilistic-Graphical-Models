# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:12:16 2019

@author: jeetp
"""

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
data = scipy.io.loadmat('soccer.mat')
game = data['game']
teamAces = game['teamAces'].tolist()
teamBruisers = game['teamBruisers'].tolist()
AcesWin = game['AcesWin'].tolist()

teamAces = np.squeeze(teamAces)
teamBruisers = np.squeeze(teamBruisers)
AcesWin = np.squeeze(AcesWin)

for i in range(0, len(AcesWin)):
    if (AcesWin[i] == -1):
        AcesWin[i] = 0;

for i in range(0, teamBruisers.shape[0]):
    for j in range(0, teamBruisers.shape[1]):
        teamBruisers[i][j] = teamBruisers[i][j] - 1

for i in range(0, teamAces.shape[0]):
    for j in range(0, teamAces.shape[1]):
        teamAces[i][j] = teamAces[i][j] - 1

np.random.seed(5)


def sigma(check):
    value_sigma = (1 / (1 + np.exp(-check)))
    return value_sigma


def joint(a, b, A):
    check1 = sigma(np.sum(a - b,axis=1))
    hmm = np.prod((check1 ** A) * (1 - check1) ** (1 - A))
    return hmm

def joint1(a, b):
    check1 = sigma(np.sum(a - b,axis=1))
    return check1

No_samples = 8000

# d=np.zeros(50)
d_sampled = np.zeros((No_samples, 40))
# b_sampled= np.zeros((No_samples,20))
# a=[0]*No_samples
# prob_check_old = np.zeros(len(d))

abk = np.zeros(40)
k = 0;
m = 0;
#while m != No_samples:
#    # while len(d_sampled)!=200:
#    #     d=np.array(d_sampled[k-1])
#    #    Prob_check_current = np.zeros(len(d))
#    for j in range(0, 40):
#        # for i in range(0, 20):
#            ddd = [-2, -1, 0, 1, 2]
#            prob = np.zeros(5)
#            abk[j] = -2;
#            check11= abk[0:20]
#            check12=abk[20:40]
#            aaa = check11[teamAces]
#            bbb = check12[teamBruisers]
#            prob[0] = joint(aaa, bbb, AcesWin)
#            abk[j] = -1;
#            check11 = abk[0:20]
#            check12 = abk[20:40]
#            aaa = check11[teamAces]
#            bbb = check12[teamBruisers]
#            prob[1] = joint(aaa, bbb, AcesWin)
#            abk[j] = 0;
#            check11 = abk[0:20]
#            check12 = abk[20:40]
#            aaa = check11[teamAces]
#            bbb = check12[teamBruisers]
#            prob[2] = joint(aaa, bbb, AcesWin)
#            abk[j] = 1;
#            check11 = abk[0:20]
#            check12 = abk[20:40]
#            aaa = check11[teamAces]
#            bbb = check12[teamBruisers]
#            prob[3] = joint(aaa, bbb, AcesWin)
#            abk[j] = 2;
#            check11 = abk[0:20]
#            check12 = abk[20:40]
#            aaa = check11[teamAces]
#            bbb = check12[teamBruisers]
#            prob[4] = joint(aaa, bbb, AcesWin)
#            #            d[i]=1
#            #            d_1= joint(d,W,b,s,p)
#            probb = np.zeros(5)
#            for ii in range(len(probb)):
#                probb[ii] = prob[ii] / np.sum(prob)
#            #            Prob_check_current[i]=prob1
#            # print(probb)
#            abk[j] = np.random.choice(np.arange(-2,3), p=probb)
#    if (k >= 25000):
#        if k % 10== 0:
#            d_sampled[m] = abk
#            m = m + 1
#    k = k + 1
# plt.plot(a);
# plt.show();
d_sampled= pickle.load(open('samples_q1.txt','rb'))
mean=np.mean(d_sampled,axis=0);
MeanA=mean[0:20]
MeanB=mean[20:40]
print(mean)
print("A best players")
print(np.argsort(MeanA)[::-1][0:10]+1)
print("B best players")
print(np.argsort(MeanB)[::-1][0:10]+1)
bbb1 = np.arange(10);
probs= []
max = 0;
output=[]
check = itertools.combinations(range(20),10)
k=0;
for value in check:
    # if(k%1000==0):
        # print(k)
    probability = 0
    # for check1 in d_sampled:
    #     hmm = check1[0:20]
    #     aaa = hmm[np.array(value)]
    #     bbb = check1[20:40][bbb1]
    #     probability += joint1(aaa,bbb)
    hmm = d_sampled[:,0:20]
    hmm1 = d_sampled[:,20:40]
    aaa = hmm[:,np.array(value)]
    bbb = hmm1[:,np.array(bbb1)]
    probability = joint1(aaa,bbb)
    probability= probability/len(d_sampled)
    probs.append(np.mean(probability))
    if np.mean(probability)>=max :
        max = np.mean(probability)
        output.append(value)
    k+=1
print(" BEst team A to play against 1 to 10 players of Team B is - ")
print(np.array(output[-1])+1)