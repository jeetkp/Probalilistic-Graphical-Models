# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:44:04 2019

@author: jeetp
"""

import scipy.io
import numpy as np
import pdb

class MessageFN():
    """ Define node structure """

    def __init__(self, var1= None, var2= None, message = None):
        self.var1 = var1;
        self.var2 = var2;
        self.message = message;
#        self.messageFN = messageFN;`

class elements():

    def __init__(self, factors=None, prob=None, message_NF = None, message_FN = None):
        # self.nodes = nodes;
        self.factors = factors;
        self.prob = prob;
        self.message_NF= np.ones((2,2))
        self.message_FN= np.ones((2,2))


factor_list= None

data = scipy.io.loadmat('pMRF.mat')
phi = data['phi']
variables = phi['variables']
table= phi['table']

variables_1 = np.array(np.squeeze(np.squeeze(np.array(np.squeeze(variables).tolist()))).tolist())
table_1 = np.array(np.squeeze(np.squeeze(np.array(np.squeeze(table).tolist()))).tolist())
neighbour_dict={};

for i in variables_1:
    neighbour_dict[i[0]]=[]
    
for i in variables_1:
    neighbour_dict[i[0]].append(i[1])
    neighbour_dict[i[1]].append(i[0])    

factortonodes = []
nodestofactors = []

bn = np.ones((2,4))

for i in range(1,5):
    neighbour= neighbour_dict[i];
    node = i;
    factors = []
    for ii in neighbour:
        factors=[]
        factors.append([i,ii])
        check = MessageFN(node,factors[0],np.ones(2));
        nodestofactors.append(check)
    
for i in variables_1:
#    neighbour= neighbour_dict[i];
    node = i;
    factors = []
    for ii in i:
        factors=[]
        factors.append([ii])
        check = MessageFN(node,factors[0],np.ones(2));
        factortonodes.append(check)

cpt=[]
for i in table_1:
    cpt.append(np.concatenate(i))
cpt= np.array(cpt)

factorlist=[]
for i , j in zip(variables_1,range(6)):
    check = elements(i,np.array(table_1[j]))
    factorlist.append(check)

threshold = 10e-6

bn = np.ones((2,4))

print("########################################################################################3")
for i in range(100):
    prev_bn = np.copy(bn)
    for ii in range(1,5):
        # prev_bn = np.copy(bn)
        # for i1 in range(len(bn[0])):
        bn[:,ii-1] = np.ones(2)
        for j1 in factorlist:
            if ii in j1.factors:
                if j1.factors[0]== ii:
                    pos1 = 0
                    pos2 = 1;
                if j1.factors[1] == ii:
                    pos1 =1;
                    pos2 =0;
                bn[:,ii-1] = bn [:,ii-1]* j1.message_FN[pos1]
                if pos1 ==1:
                    check1= np.matmul(np.transpose(j1.prob),j1.message_NF[pos2])
                else:
                    check1= np.matmul(j1.prob, j1.message_NF[pos2])
                check2 = np.ones(2);
                for j2 in factorlist:
                    if (ii in j2.factors) and (j1.factors[pos2] not in j2.factors):
                        if j2.factors[0] == ii:
                            pos11= 0
                        else:
                            pos11 = 1;
                        check2 = check2 * j2.message_FN[pos11]
                j1.message_FN[pos1] = check1
                j1.message_NF[pos1] = check2
    if np.all(np.absolute(prev_bn - bn) < threshold) and i > 1:
        print("Convergence in these many iterations - ")
        print(i)
        # print(bn)
        break

bn = bn/np.sum(bn,axis=0)
print("Belief after Loopy belief Propagation - ")
print(bn)
print("#######################################################################################################")

probs = np.ones(16)
pi = np.zeros((2,4))
count11= np.zeros((2,4))
for i in range(2**4):
    i_bin = bin(int(i))[2:]
    i_bin = i_bin.zfill(4);
    i_bin = list(i_bin)

    for ii,jj in zip(variables_1,range(6)):
        x = int(''.join(str(i_bin[x-1]) for x in ii), 2);
        probs[i] = probs[i] * cpt[jj][x]
    for i2,j2 in zip(i_bin,range(4)):
        if i2=='0':
            pi[0][j2]+=probs[i]
            count11[0][j2] += 1
        else:
            pi[1][j2] += probs[i]
            count11[1][j2] += 1


pi[:,0]=pi[:,0]/np.sum(pi[:,0])
pi[:,1]=pi[:,1]/np.sum(pi[:,1])
pi[:,2]=pi[:,2]/np.sum(pi[:,2])
pi[:,3] = pi[:,3]/np.sum(pi[:,3])
print("Probabilties after Enumeration - ")
print(pi)

print(" Mean error difference between the Loopy Belief and Enumeration-  ")
print(np.mean(np.abs(bn-pi)))
    