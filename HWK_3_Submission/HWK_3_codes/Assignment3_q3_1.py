# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:44:04 2019

@author: jeetp
"""

import scipy.io
import numpy as np


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

processed = np.zeros(4)
q = np.zeros(4)
q1=np.ones(4)
q= np.vstack((q,q1))
cpt=[]
for i in table_1:
    cpt.append(np.concatenate(i))
cpt= np.array(cpt)
k=4
count = 0
while np.count_nonzero(processed) != 4:
    check= 0;
    prob = np.zeros(2);
    prob1 = np.zeros(2);
    val = np.zeros(2)
    # k=np.where(processed == 0)[0][0]+1
    if processed[k-1]==0:
        neighbours = neighbour_dict[k];
        for i in neighbours:
            for i1,j1 in zip(variables_1,range(5)):
                if i1[0]==i:
                    if i1[1]==k:
                        index = (j1)
                        val=i1;
                        break
                if i1[0]==k:
                    if i1[1]==i:
                        index = (j1)
                        val=i1
                        break
            for ii in range(2):
                if val[0] == k:
                    prob[0]= cpt[index][0]
                    prob[1]= cpt[index][2]
                    prob1[0] = cpt[index][1]
                    prob1[1] = cpt[index][3]
                else:
                    prob[0]= cpt[index][0]
                    prob[1]= cpt[index][1]
                    prob1[0] = cpt[index][2]
                    prob1[1] = cpt[index][3]
                if ii == 0:
                    check = check + (q[ii][i-1]*np.log(prob))
                else:
                    check = check + (q[ii][i - 1] * np.log(prob1))
        check = np.exp(check)
        sum1=np.sum(check);
        check[0] = check[0]/sum1
        check[1] = check[1] / sum1
        processed[k - 1] = 1;
       # if np.abs(q[:,k-1][0]-check[0])>10e-6 :
#         q[:,k-1]=check
#         processed[k-1]=1;
        if(q[:,k-1][0]!=check[0]):
            for i in neighbours:
                processed[i-1]=0
        q[:, k - 1] = check
#        print(processed)
        count+=1;
#        print(q)
    else:
        if k+1 >4:
            k=1;
        else:
            k+=1
print("Data after mean field - ")
print(q)
#print(np.prod(q,axis=1))
#print(count)
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
#print(probs)
#print(sum(probs))
#print("#######################")
#print(pi)
# print(count11)
pi[:,0]=pi[:,0]/np.sum(pi[:,0])
pi[:,1]=pi[:,1]/np.sum(pi[:,1])
pi[:,2]=pi[:,2]/np.sum(pi[:,2])
pi[:,3] = pi[:,3]/np.sum(pi[:,3])
print("Data after Enumeration and marginalizing - ")
print(pi)

print(" Mean Error Difference between Mean field and enumeration.- ")
print(np.mean(np.abs(pi-q))) 