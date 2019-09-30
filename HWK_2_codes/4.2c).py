import numpy as np
import random
import time

class Factors():
    """To define FACTORS FOR  variable elimination"""
    def __init__(self,nodes=None,table=None):
        self.nodes=nodes;
        self.table=table;

    def eliminate(self,nodes,table,variable):
        factors=[]
        prob=[]
        node_backup=nodes;
        check=[]
        for i in range(0,len(node_backup)):
            for j in range(0,len(node_backup[i])):
                if(node_backup[i][j]==variable):
                    factors.append(node_backup[i]);
                    prob.append(table[i]);
                    check.append(i)
                    # nodes.pop(i)
        count=0;
        for i in check:
            index=i-count;
            nodes.pop(index);
            table.pop(index);
            count=count+1;

        while(len(factors)!=1):
            # factors_1=[factors[0],factors[1]]
            # probb_1=[prob[0],prob[1]]
            new_factors,new_prob= self.multiply(factors,prob);
            factors=new_factors
            prob=new_prob

        new_factors, new_prob = self.reduce(factors,prob,variable);
        # factors.pop(0);
        # factors.pop(0);
        # factors.append(new_factors);
        # prob.pop(0);
        # prob.pop(0);
        nodes.append(new_factors[0]);
        table.append(new_prob)
        return nodes,table

    def multiply(self,factors,prob):
        new_factors = []
        new_probbb = []
        for i in range(0,len(factors),2):
            if i+1==len(factors):
                new_factors.append(factors[i])
                new_probbb.append(prob[i])
                return new_factors,new_probbb
            n=len(factors[i])
            index1=[]
            prob1=[]
            for ii in range(0,2**n):
                i_bin = bin(ii)[2:]
                i_bin = i_bin.zfill(n);
                i_bin = list(i_bin)
                index1.append(i_bin);
                prob1.append(prob[i][ii])
        # for i in range(0, len(factors[i+1])):
            n1 = len(factors[i+1])
            index2 = []
            prob2= []
            for ii in range(0, 2** n1):
                i_bin = bin(ii)[2:]
                i_bin = i_bin.zfill(n1);
                i_bin = list(i_bin)
                index2.append(i_bin);
                prob2.append(prob[i+1][ii])
            # combo= union(factors[i],factors[i+1]);
            combo= list(set(factors[i]) | set(factors[i+1]));
            # common= intersect(factors[i],factors[i+1]);
            common = list(set(factors[i]) & set(factors[i+1]))
            size1= len(combo);
            probbbb = []
            for ii in range(0,2**size1):

                probb =1;
                i_bin = bin(ii)[2:]
                i_bin = i_bin.zfill(size1);
                i_bin = list(i_bin);
                checkk = []
                checkk1 = []
                for iii in factors[i]:
                    for jjj in range(0,len(combo)):
                        if(combo[jjj]== iii):
                            checkk.append(i_bin[jjj])
                            bin_str = ''.join(str(e) for e in checkk)
                            check11 = int(bin_str, 2);
                for iii in factors[i+1]:
                    for jjj in range(0,len(combo)):
                        if(combo[jjj]== iii):
                            checkk1.append(i_bin[jjj])
                            bin_str1 = ''.join(str(e) for e in checkk1)
                            check22 = int(bin_str1, 2);
                # for iiii in range(0,len(index1)):
                #     if list(set(index1[iiii]) | set(checkk))== checkk:
                probb=probb*prob1[check11]
                # for iiii in range(0, len(index2)):
                #     if list(set(index2[iiii]) | set(checkk1)) == checkk1:
                probb = probb * prob2[check22]

                probbbb.append(probb);
            new_factors.append(combo);
            new_probbb.append(probbbb)
        return new_factors,new_probbb

    def reduce(self, factors, prob, variable):
        list1=[]
        probb=[]
        n=len(factors[0]);
        # pos=0;
        for i in range(0,len(factors[0])):
            if factors[0][i]==variable:
                pos=i;
        n=n-1;
        for i in range(0, 2 ** (n)):
            i_bin = bin(i)[2:]
            i_bin_1 = bin(i)[2:]
            i_bin = i_bin.zfill(n);
            i_bin_1 = i_bin_1.zfill(n);
            i_bin = list(i_bin)
            i_bin_1 = list(i_bin_1)
            # i_bin_1= i_bin;
            i_bin.insert(pos, '0');
            bin_str1 = ''.join(str(e) for e in i_bin)
            snap = int(bin_str1, 2);
            i_bin_1.insert(pos,'1');
            bin_str11 = ''.join(str(e) for e in i_bin_1)
            snap1 = int(bin_str11, 2);
            list1.append(snap);
            list1.append(snap1);
        for i in range(0,len(list1),2):
            probability=prob[0][list1[i]]+prob[0][list1[i+1]]
            probb.append(probability);
        factors[0].remove(variable);
        return factors,probb

class Node_element():
    """ Define node structure """

    def __init__(self, numId=None, values=None, parents=None, children=None, peye=None, lam=None, prob=None, p_back=None):
        self.numId = numId;
        self.values = values;
        self.parents = parents;
        self.children = children;
        self.peye = peye;
        self.lam = lam;
        self.prob = prob;
        self.prob_back = p_back;

class BayesianGraphicalModel():
    """Creates a baysian network of the nodes passed"""

    def func(self, adj, values, Cpt):
        N = np.shape(adj)[0]
        group=[]
        adj_trap= adj.transpose();
        root=[]
        for i in range(0,np.shape(adj)[0]):
            if sum(adj_trap[i])==0:
                root.append(i);
        for i in range(0,np.shape(adj)[0]):
            numid=i;
            values= 0;
            parents= (np.nonzero(adj[:,i]));
            parents= parents[0].tolist();
            children = (np.nonzero(adj[i,:]));
            children = children[0].tolist();
            cpt=Cpt[i];
            group.append(Node_element(numid,values,parents,children,None,None,cpt,None));
        return group,root


    def jointDistri(self,group,root,test):
        # test.reverse();
        parent_states=test;
        joint=1;
        for nodes in group:
            if nodes.numId in root:
                state= parent_states[nodes.numId];
                joint = joint * nodes.prob[0][int(state)];
            else:
                state= parent_states[nodes.numId];
                parents=nodes.parents;
                parent_state=[]
                j=1
                for val in parents:
                    parent_state.append(parent_states[val])
                # for val in parent_state:
                #     j = j* nodes.prob[val][state];
                bin_str= ''.join(str(e) for e in parent_state)
                check=int(bin_str,2);
                # print(check)
                joint= joint * nodes.prob[check][int(state)];
        return(joint);

    def accuracy_check(self):
        diff = []
        # with open('C:/Users/jeetp/OneDrive/Desktop/GAtech study material/ece8803 PGM/Assignment2/joint.dat', 'r') as f:
        with open('joint.dat', 'r') as f:
            d = f.readlines()
            for i in d:
                k = i.split('\t')
                i_bin = bin(int(k[0]))[2:]
                i_bin = i_bin.zfill(12);
                i_bin = list(i_bin)
                i_bin.reverse();
                val = GrapMdel.jointDistri(group, root, i_bin)
                val_given = k[1].split('\n');
                diff.append(abs(val - float(val_given[0])));
        return diff

    # def get_data(self,filedata,observed_var,observed_var_state,query_var,query_var_state):
    #     count = 0;
    #     count1 = 0;
    #     query_var_states_1=[]
    #     prob = []
    #     for i in range(0,2**len(query_var)):
    #         i_bin = bin(int(i))[2:]
    #         i_bin = i_bin.zfill(len(query_var));
    #         i_bin= list(i_bin)
    #         query_var_states_1.append(i_bin);
    #     for query_var_1 in query_var_states_1:
    #         count=0;
    #         count1=0;
    #         for data in filedata:
    #             flag=0;
    #             flag1=0;
    #             for i in range (0,len(observed_var)):
    #                 if(data[observed_var[i]] == observed_var_state[i]):
    #                     flag = 1;
    #                 else:
    #                     flag=0;
    #             if flag == 1:
    #                 count=count+1;
    #                 for j in range (0,len(query_var)):
    #                     if (data[int(query_var[j])] == query_var_1[j]):
    #                         flag1 = 1;
    #                 if flag1  ==1:
    #                    count1 = count1+1;
    #         prob.append(count1/count);
    #     return(prob)
    #
    # def get_data_1(self,observed_var,observed_var_state,query_var,query_var_state):
    #     count = 0.0;
    #     count1 = 0.0;
    #     query_var_states_1=[]
    #     prob = []
    #     for i in range(0,2**len(query_var)):
    #         i_bin = bin(int(i))[2:]
    #         i_bin = i_bin.zfill(len(query_var));
    #         i_bin= list(i_bin)
    #         query_var_states_1.append(i_bin);
    #     for query_var_1 in query_var_states_1:
    #         count=0.0;
    #         count1=0.0;
    #         with open('C:/Users/jeetp/OneDrive/Desktop/GAtech study material/ece8803 PGM/Assignment2/joint.dat','r') as f:
    #             d = f.readlines()
    #             for i in d:
    #                 k = i.split('\t')
    #                 i_bin = bin(int(k[0]))[2:]
    #                 i_bin = i_bin.zfill(12);
    #                 i_bin = list(i_bin)
    #                 i_bin.reverse();
    #                 # val = GrapMdel.jointDistri(group, root, i_bin)
    #                 val_given = k[1].split('\n');
    #                 # diff.append(abs(val - float(val_given[0])));
    #         # for data in filedata:
    #         #     flag=0;
    #         #     flag1=0;
    #                 flag=0
    #                 flag1=0
    #                 for ii in range (0,len(observed_var)):
    #                     if(i_bin[observed_var[ii]] == observed_var_state[ii]):
    #                         flag = 1;
    #                     else:
    #                         flag=0;
    #                 if flag == 1:
    #                     count=count+float(val_given[0]);
    #                     for j in range (0,len(query_var)):
    #                         if (i_bin[int(query_var[j])] == query_var_1[j]):
    #                             flag1 = 1;
    #                     if flag1  ==1:
    #                        count1 = count1+float(val_given[0]);
    #         prob.append(count1/count);
    #     return(prob)

    def get_data_1(self,observed_var,observed_var_state,query_var,query_var_state):
        count = 0.0;
        count1 = 0.0;
        query_var_states_1=[]
        prob = []
        for i in range(0,2**len(query_var)):
            i_bin = bin(int(i))[2:]
            i_bin = i_bin.zfill(len(query_var));
            i_bin= list(i_bin)
            query_var_states_1.append(i_bin);
        for query_var_1 in query_var_states_1:
            count=0.0;
            count1=0.0;
            # with open('C:/Users/jeetp/OneDrive/Desktop/GAtech study material/ece8803 PGM/Assignment2/joint.dat','r') as f:
            with open('joint.dat','r') as f:
                d = f.readlines()
                for i in d:
                    k = i.split('\t')
                    i_bin = bin(int(k[0]))[2:]
                    i_bin = i_bin.zfill(12);
                    i_bin = list(i_bin)
                    i_bin.reverse();
                    # val = GrapMdel.jointDistri(group, root, i_bin)
                    val_given = k[1].split('\n');
                    # diff.append(abs(val - float(val_given[0])));
            # for data in filedata:
            #     flag=0;
            #     flag1=0;
                    flag=0
                    flag1=0
                    for ii in range (0,len(observed_var)):
                        if(i_bin[observed_var[ii]] == observed_var_state[ii]):
                            flag = 1;
                        else:
                            flag=0;
                    if flag == 1:
                        count=count+float(val_given[0]);
                        for j in range (0,len(query_var)):
                            if (i_bin[int(query_var[j])] == query_var_1[j]):
                                flag1 = 1;
                        if flag1  ==1:
                           count1 = count1+float(val_given[0]);
            prob.append(count1/count);
        return(prob)

    def get_data_11(self,filedata_1,filedata_1_count,observed_var,observed_var_state,query_var,query_var_state):
        count = 0.0;
        count1 = 0.0;
        query_var_states_1=[]
        prob = []
        for i in range(0,2**len(query_var)):
            i_bin = bin(int(i))[2:]
            i_bin = i_bin.zfill(len(query_var));
            i_bin= list(i_bin)
            query_var_states_1.append(i_bin);
        for query_var_1 in query_var_states_1:
            count=0.0;
            count1=0.0;
            for d in filedata_1:
                # for i in d:
                #     k = filedata_1_count[d]
                    i_bin = bin(int(d))[2:]
                    i_bin = i_bin.zfill(12);
                    i_bin = list(i_bin)
                    i_bin.reverse();
                    # val = GrapMdel.jointDistri(group, root, i_bin)
                    val_given = (filedata_1_count[d])/sum(filedata_1_count)
                    # diff.append(abs(val - float(val_given[0])));
            # for data in filedata:
            #     flag=0;
            #     flag1=0;
                    flag=0
                    flag1=0
                    for ii in range (0,len(observed_var)):
                        if(i_bin[observed_var[ii]] == observed_var_state[ii]):
                            flag = 1;
                        else:
                            flag=0;
                    if flag == 1:
                        count=count+float(val_given);
                        for j in range (0,len(query_var)):
                            if (i_bin[int(query_var[j])] == query_var_1[j]):
                                flag1 = 1;
                        if flag1  ==1:
                           count1 = count1+float(val_given);
            prob.append(count1/count);
        return(prob)

    # def get_data(self,filedata,observed_var,observed_var_state,query_var,query_var_state):
    #     count = 0;
    #     count1 = 0;
    #     query_var_states_1=[]
    #     prob = []
    #     for i in range(0,2**len(query_var)):
    #         i_bin = bin(int(i))[2:]
    #         i_bin = i_bin.zfill(len(query_var));
    #         i_bin= list(i_bin)
    #         query_var_states_1.append(i_bin);
    #     for query_var in query_var_states_1:
    #         count=0;
    #         count1=0;
    #         for data in filedata:
    #             flag=0;
    #             flag1=0;
    #             for i in range (0,len(observed_var)):
    #                 if(data[observed_var[i]] == observed_var_state[i]):
    #                     flag = 1;
    #             if flag == 1:
    #                 count=count+1;
    #                 for j in range (0,len(query_var)):
    #                     if (data[query_var[j]] == query_var_state[j]):
    #                         flag1 = 1;
    #                 if flag1  ==1:
    #                    count1 = count1+1;
    #             prob.append(count1/count);
    #     return(prob)

    def topological_sort_temp(self,group,i,v,s):
        v[i] = 1
        for j in group[i].children:
            if v[j]==0:
                self.topological_sort_temp(group,j,v,s);
        s.insert(0,i)

    def topological_sort(self,group):
        num=[0]*12;
        s=[];
        for i in range(12):
            if num[i]== 0 :
                self.topological_sort_temp(group,i,num,s)
        return(s)

    # def fwd_Sam(self,group,topological_order,N):
    #     sample=[];
    #     for i in range(N):
    #         data=[0]*12;
    #         for j in topological_order:
    #             pa=group[j].parents;
    #             bi=0;
    #             if(len(pa)):
    #                 bi=int(''.join(str(data[x]) for x in pa),2);
    #                 print(group[j].prob[bi][1])
    #                 data[j]=np.random.binomial(1,group[j].prob[bi][1]);
    #         d=int(''.join(str(x) for x in data),2);
    #         sample.append(d);
    #     return sample

    def fwd_Sam_1(self,group,topological_order,root):
        # sample=[];
        # for i in range(N):
            data=[0]*12;
            for j in topological_order:
                if j in root:
                    data[j]=np.random.binomial(1,group[j].prob[0][1]);
                else:
                    parent_check=group[j].parents;
                    parent_state=[]
                    for val in parent_check:
                        parent_state.append(data[val])
                    val=int(''.join(str(x) for x in parent_state),2);
                    data[j]=np.random.binomial(1,group[j].prob[val][1]);
                    bin_str = ''.join(str(e) for e in data)
                    check = int(bin_str, 2);
            # sample.append(d);
            return check



if __name__ == "__main__":
    # print(time.time())
    adjmat = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]);

    cpt = {};
    filedata=[]
    # with open('C:/Users/jeetp/OneDrive/Desktop/GAtech study material/ece8803 PGM/Assignment2/dataset.dat', 'r') as f:
    #     d = f.readlines()
    #     for i in d:
    #         k = i.split('\n')
    #         i_bin = bin(int(k[0]))[2:]
    #         i_bin = i_bin.zfill(12);
    #         # print()
    #         i_bin = list(i_bin);
    #         i_bin.reverse();
    #         filedata.append(i_bin)
    adjmat_transpose = adjmat.transpose();
    filedata_1 =[]
    filedata_1_count = []
    for i in range(0,2**(len(adjmat[0]))):
        filedata_1.append(i);
        filedata_1_count.append(0);
    # with open('C:/Users/jeetp/OneDrive/Desktop/GAtech study material/ece8803 PGM/Assignment2/dataset.dat', 'r') as f:
    with open('dataset.dat','r') as f:
        d = f.readlines()
        for i in d:
            k = i.split('\n')
            filedata_1_count[int(k[0])]= filedata_1_count[int(k[0])] +1;

    # print(filedata_1)
    # print(filedata_1_count)





    for i in range(0, np.shape(adjmat)[0]):
        N = np.count_nonzero(adjmat_transpose[i]);
        if N == 0:
            check1= []
            check = random.random();
            check1.append([check, 1 - check])
            cpt[i] = check1;
        else:
            check1 = []
            for j in range(0, 2**N):
                check = random.random();
                check1.append([check, 1 - check])
            cpt[i] = check1;
    # for data in
    # print(len(filedata))
    # for i in filedata:
    #     i.reverse();
    values = 0;
    GrapMdel=BayesianGraphicalModel();
    [group, root] = GrapMdel.func(adjmat, values,cpt);
    # I have run this to get the model of my graph. Now i will update the Conditional probability.
    # cpt1={};
    # for node in group:
    #     if node.numId in root:
    #         n=0;
    #         for i in filedata:
    #             if i[int(node.numId)]=='0':
    #                 n= n+1;
    #         node.prob[0][0]=n/len(filedata);
    #         node.prob[0][1]=1-node.prob[0][0]
    #         cpt1[node.numId]= ([node.prob[0][0],node.prob[0][1]])
    #     else:
    #         check1=[]
    #         parents_data=[]
    #         for values in node.parents:
    #             parents_data.append(values);
    #         n=len(parents_data)
    #         for i in range(0,(2**n)):
    #             count=0;
    #             count1=0;
    #             num= bin(i)[2:]
    #             num=num.zfill(n);
    #             num=list(num);
    #             for j in filedata:
    #                 flag=0;
    #                 for k in range(0,n):
    #                     if num[k]==j[parents_data[k]]:
    #                         flag=1;
    #                     else:
    #                         flag=0;
    #                 if flag ==1:
    #                     count=count+1;
    #                     # flag = 0;
    #                 # for k in range(0, n ):
    #                     # if num[k] == j[parents_data[k]]:
    #                         # flag = 1;
    #                 if flag ==1 and j[node.numId]=='0':
    #                     count1 = count1 +1;
    #             node.prob[i][0] = count1 / count;
    #             node.prob[i][1] = 1 - node.prob[i][0]
    #             check1.append([node.prob[i][0],node.prob[i][1]])
    #         cpt1[node.numId]=(check1);
    # print(time.time())
    ##################################################################################################################
    #Calculate CPTS

    cpt1 = {};
    for node in group:
        if node.numId in root:
            n = 0;
            for i in filedata_1:
                i_bin = bin(i)[2:]
                i_bin = i_bin.zfill(12);
                i_bin = list(i_bin);
                i_bin.reverse();
                if i_bin[int(node.numId)] == '0':
                    n = n + filedata_1_count[i];
                    # n= n * filedata_1_count(i);
            node.prob[0][0] = n / sum(filedata_1_count);
            node.prob[0][1] = 1 - node.prob[0][0]
            cpt1[node.numId] = ([node.prob[0][0], node.prob[0][1]])
        else:
            check1 = []
            parents_data = []
            for values in node.parents:
                parents_data.append(values);
            n = len(parents_data)
            for i in range(0, (2 ** n)):
                count = 0;
                count1 = 0;
                num = bin(i)[2:]
                num = num.zfill(n);
                num = list(num);
                for j in filedata_1:
                    i_bin = bin(j)[2:]
                    i_bin = i_bin.zfill(12);
                    i_bin = list(i_bin);
                    i_bin.reverse();
                    flag = 0;
                    parent_check=[]
                    for kkk in range(0,n):
                        parent_check.append(i_bin[parents_data[kkk]]);
                    d = int(''.join(str(x) for x in parent_check), 2);
                    if d==i:
                        flag=1;
                    # for k in range(0, n):
                    #     if num[k] == i_bin[parents_data[k]]:
                    #         flag = 1;
                    #     else:
                    #         flag = 0;
                    if flag == 1:
                        count = count +  filedata_1_count[j];
                        # flag = 0;
                    # for k in range(0, n ):
                    # if num[k] == j[parents_data[k]]:
                    # flag = 1;
                    if flag == 1 and i_bin[node.numId] == '0':
                        count1 = count1 +  filedata_1_count[j];
                node.prob[i][0] = count1 / count;
                node.prob[i][1] = 1 - node.prob[i][0]
                check1.append([node.prob[i][0], node.prob[i][1]])
            cpt1[node.numId] = (check1);
    ##################################################################################################################

    # diff=[];
    diff = GrapMdel.accuracy_check();
    # with open('C:/Users/jeetp/OneDrive/Desktop/GAtech study material/ece8803 PGM/Assignment2/joint.dat', 'r') as f:
    #     d = f.readlines()
    #     for i in d:
    #         k = i.split('\t')
    #         i_bin = bin(int(k[0]))[2:]
    #         i_bin = i_bin.zfill(12);
    #         i_bin = list(i_bin)
    #         i_bin.reverse();
    #         val = GrapMdel.jointDistri(group,root,i_bin)
    #         val_given = k[1].split('\n');
    #         diff.append(abs(val - float(val_given[0])));
    print("Accuracy of the model is ")
    print(sum(diff))
    query_var_state = []
    #######################################################################################
    nodes=[]
    table=[]
    for node in group:
        check1=[]#count for parents
        check2=[]#for probabilities
        n= len(node.parents);
        prob=node.prob;
        if node.numId in root:
            check1.append(node.numId);
            for i in range(0,len(prob[0])):
                for j in range(0,len(prob)):
                    check2.append(prob[j][i]);
        else:
            check1.append(node.numId);
            for n in node.parents:
                check1.append(n);
            for i in range(0,len(prob[0])):
                for j in range(0,len(prob)):
                    check2.append(prob[j][i]);
        nodes.append(check1)
        table.append(check2);

    FactorModel = Factors(nodes,table);
    ##################################################################
    present=time.time()
    print(" Prob of Observed Variables: HasFever=true, Coughs=true; Query Variable: HasFlu")
    observed_var = [8, 11];
    query_var = [1]
    deletes=[]
    hmm=[0,1,2,3,4,5,6,7,8,9,10,11]
    for j in observed_var:
        hmm.remove(j);
    for j in query_var:
        hmm.remove(j)
    # for j in observed_var:
    #     for i in range(0,len(hmm)):
    #         if j==hmm[i]:
    #             deletes.append(i);
    # for i in deletes:
    #     hmm.remove(i)
    # deletes=[]
    # for j in query_var:
    #     for i in range(0,len(hmm)):
    #         if j==hmm[i]:
    #             deletes.append(i);
    # for i in deletes:
    #     hmm.remove(i)
    hmm.reverse();
    for i in hmm:
        nodes_new,table_table = FactorModel.eliminate(nodes,table,i)
        nodes=nodes_new;
        table=table_table;

    while (len(nodes) != 1):
        nodes_new, table_table = FactorModel.multiply(nodes, table)
        nodes = nodes_new
        table = table_table
    observed_var_state = ['1', '1'];
    n=len(nodes[0]);
    data=[]
    prob=[]
    pos=[]
    for j in observed_var:
        for i in range(0,n):
            if j==nodes[0][i]:
                pos.append(i);
    for i in range(0,2**n):
        i_bin = bin(i)[2:]
        i_bin = i_bin.zfill(n);
        i_bin = list(i_bin)
        flag=True;
        for ii in range(0,len(pos)):
            if i_bin[pos[ii]]!=observed_var_state[ii]:
                flag=False;
        if flag==True:
            data.append(i_bin);
            prob.append(table[0][i])

    for i in range(0,len(data)):
        print(data[i]);
        print(prob[i]/sum(prob));

    end= time.time();
    print("TIme Taken is ")
    print(end-present)

    prob_query = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    print("Joint model queries Variables: HasFever=true, Coughs=true; Query Variable: HasFlu")
    print(prob_query)
    #######################################################################################
    nodes = []
    table = []
    for node in group:
        check1 = []  # count for parents
        check2 = []  # for probabilities
        n = len(node.parents);
        prob = node.prob;
        if node.numId in root:
            check1.append(node.numId);
            for i in range(0, len(prob[0])):
                for j in range(0, len(prob)):
                    check2.append(prob[j][i]);
        else:
            check1.append(node.numId);
            for n in node.parents:
                check1.append(n);
            for i in range(0, len(prob[0])):
                for j in range(0, len(prob)):
                    check2.append(prob[j][i]);
        nodes.append(check1)
        table.append(check2);

    FactorModel = Factors(nodes, table);
    ###########################################################################
    present=time.time();
    print(" Prob OF HasRash, Coughs, IsFatigued, Vomits,and HasFever) given the patient has pneumonia")
    observed_var = [4];
    query_var = [7,8,9,10,11]
    deletes = []
    hmm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for j in query_var:
        hmm.remove(j);
    for j in observed_var:
        hmm.remove(j);
    # for j in observed_var:
    #     for i in range(0, len(hmm)):
    #         if j == hmm[i]:
    #             deletes.append(i);
    # for i in deletes:
    #     hmm.remove(i)
    # deletes = []
    # for j in query_var:
    #     for i in range(0, len(hmm)):
    #         if j == hmm[i]:
    #             deletes.append(i);
    # for i in deletes:
    #     hmm.remove(i)

    for i in hmm:
        nodes_new, table_table = FactorModel.eliminate(nodes, table, i)
        nodes = nodes_new;
        table = table_table;

    while (len(nodes) != 1):
        nodes_new, table_table = FactorModel.multiply(nodes, table)
        nodes = nodes_new
        table = table_table

    observed_var_state = ['1'];
    n = len(nodes[0]);
    data = []
    prob = []
    pos = []
    for j in observed_var:
        for i in range(0, n):
            if j == nodes[0][i]:
                pos.append(i);
    for i in range(0, 2 ** n):
        i_bin = bin(i)[2:]
        i_bin = i_bin.zfill(n);
        i_bin = list(i_bin)
        flag = True;
        for ii in range(0, len(pos)):
            if i_bin[pos[ii]] != observed_var_state[ii]:
                flag = False;
        if flag == True:
            data.append(i_bin);
            prob.append(table[0][i])

    for i in range(0, len(data)):
        print(data[i]);
        print(1-(prob[i]/sum(prob)));

    end = time.time();
    print("TIme Taken is ")
    print(end - present)

    prob_query = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    print("Joint model queries (HasRash, Coughs, IsFatigued, Vomits, and HasFever) given the patient has pneumonia")
    print(prob_query)
    ##########################################################################################
    nodes = []
    table = []
    for node in group:
        check1 = []  # count for parents
        check2 = []  # for probabilities
        n = len(node.parents);
        prob = node.prob;
        if node.numId in root:
            check1.append(node.numId);
            for i in range(0, len(prob[0])):
                for j in range(0, len(prob)):
                    check2.append(prob[j][i]);
        else:
            check1.append(node.numId);
            for n in node.parents:
                check1.append(n);
            for i in range(0, len(prob[0])):
                for j in range(0, len(prob)):
                    check2.append(prob[j][i]);
        nodes.append(check1)
        table.append(check2);

    FactorModel = Factors(nodes, table);
    #############################################################################################################
    present=time.time()
    observed_var = [0];
    query_var = [10]
    deletes = []
    hmm = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    for j in observed_var:
        hmm.remove(j);
    for j in query_var:
        hmm.remove(j);

    for i in hmm:
        nodes_new, table_table = FactorModel.eliminate(nodes, table, i)
        nodes = nodes_new;
        table = table_table;

    while(len(nodes)!=1):
        nodes_new,table_table=FactorModel.multiply(nodes,table)
        nodes=nodes_new
        table=table_table
    observed_var_state = ['1'];
    n = len(nodes[0]);
    data = []
    prob = []
    pos = []
    for j in observed_var:
        for i in range(0, n):
            if j == nodes[0][i]:
                pos.append(i);
    for i in range(0, 2 ** n):
        i_bin = bin(i)[2:]
        i_bin = i_bin.zfill(n);
        i_bin = list(i_bin)
        flag = True;
        for ii in range(0, len(pos)):
            if i_bin[pos[ii]] != observed_var_state[ii]:
                flag = False;
        if flag == True:
            data.append(i_bin);
            prob.append(table[0][i])

    for i in range(0, len(data)):
        print(data[i]);
        print(prob[i]/sum(prob));

    end = time.time();
    print("TIme Taken is ")
    print(end - present)
    prob_query = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    print("Joint model queries probability of vomitting in summer?")
    print(prob_query)
    ##########################################################################################
    # prob_query= GrapMdel.get_data(filedata,observed_var,observed_var_state,query_var,query_var_state)
    # prob_query_1 = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    # print(prob_query)
    #
    # observed_var = [4];
    # observed_var_state = ['1'];
    # query_var = [7,8,9,10,11]
    # query_var_state = ['1','1','1','1','1'];
    # prob_query = GrapMdel.get_data(filedata, observed_var, observed_var_state, query_var, query_var_state)
    # prob_query_1 = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    # print(prob_query)
    #
    # observed_var = [0];
    # observed_var_state = ['1'];
    # query_var = [10]
    # query_var_state = ['1'];
    # prob_query = GrapMdel.get_data(filedata, observed_var, observed_var_state, query_var, query_var_state)
    # prob_query_1 = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    # print(prob_query)
    #
    # topological_order= GrapMdel.topological_sort(group);
    # dataset= GrapMdel.fwd_Sam(group,topological_order,100)
    #
    # N=1000;
    # dataset=[]
    # for i in range(0,1000):
    #     dataset.append(GrapMdel.fwd_Sam_1(group,topological_order,root));
    #
    # print(dataset);
    #
    # test= [1,1,1,1,1,1,1,1,1,1,1,0]
    # test.reverse();
    # val= GrapMdel.jointDistri(group,root,test);
    # print(val)
    # observed_var = [8, 11];
    # observed_var_state = ['1', '1'];
    # query_var = [1]
    # query_var_state = ['1'];
    # # prob_query= GrapMdel.get_data(filedata_1,filedata_1_count,observed_var,observed_var_state,query_var,query_var_state)
    # prob_query = GrapMdel.get_data_11(filedata_1, filedata_1_count, observed_var, observed_var_state, query_var,query_var_state)
    # print("My model queries Observed Variables: HasFever=true, Coughs=true; Query Variable: HasFlu")
    # print(prob_query)
    # prob_query = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    # print("Joint model queries Variables: HasFever=true, Coughs=true; Query Variable: HasFlu")
    # print(prob_query)
    #
    # observed_var = [4];
    # observed_var_state = ['1'];
    # query_var = [7, 8, 9, 10, 11]
    # query_var_state = ['1', '1', '1', '1', '1'];
    # prob_query = GrapMdel.get_data_11(filedata_1, filedata_1_count, observed_var, observed_var_state, query_var,
    #                                   query_var_state)
    # print("My model queries (HasRash, Coughs, IsFatigued, Vomits,and HasFever) given the patient has pneumonia")
    # print(prob_query)
    # prob_query = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    # print("Joint model queries (HasRash, Coughs, IsFatigued, Vomits, and HasFever) given the patient has pneumonia")
    # print(prob_query)
    #
    # observed_var = [0];
    # observed_var_state = ['1'];
    # query_var = [10]
    # query_var_state = ['1'];
    # prob_query = GrapMdel.get_data_11(filedata_1, filedata_1_count, observed_var, observed_var_state, query_var,
    #                                   query_var_state)
    # print("My model queries probability of vomitting in summer?")
    # print(prob_query)
    # prob_query = GrapMdel.get_data_1(observed_var, observed_var_state, query_var, query_var_state)
    # print("Joint model queries probability of vomitting in summer?")
    # print(prob_query)
