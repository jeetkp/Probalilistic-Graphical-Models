import numpy as np
import random

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
        test.reverse();
        parent_states=test;
        joint=1;
        for nodes in group:
            if nodes.numId in root:
                state= parent_states[nodes.numId];
                joint = joint * nodes.prob[0][state];
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
                joint= joint * nodes.prob[check][state];
        return(joint);








if __name__ == "__main__":
    adjmat = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]);

    cpt = {};
    adjmat_transpose = adjmat.transpose();
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
    values = 0;
    GrapMdel=BayesianGraphicalModel();
    [group, root] = GrapMdel.func(adjmat, values,cpt);
    test= [1,1,1,1,1,1,1,1,1,1,1,0]
    val= GrapMdel.jointDistri(group,root,test);
    print(val)
