In this code the basic network is being defined with all the data given of estimated and true probabilities. 

This has a bayesian network graph, which does all initialization of all data relating to probabilities in the nodes that we defined and also queries and samples them as well. 

There is a class called node_element which is basically a structure of the node. 
The forward sampling function generates random dataset and the likelihood to it is also calculated. 

All data asked in the question gets prinited. I have attached a screenshot of the data as well in the report. 

The code is very effiecient and runs within seconds. 

As of now the query variables are set as the ones given in the question. 
observed_var = [8,11];
observed_var_state = ['1','1'];
query_var = [1]

WE can update accordingly and the code will send back the appropriate estimated probability and true probability. 

The model is stored in the form of adjacency matrix so every time we run the code, we redefine the probabilities if there are any changes in adjacency matrix. 
Ex- 
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


The datasets 'dataset.dat' and 'joint.dat' are mandatory required in the same folder as code, since we keep accessing ths files for the data required. 

Numpy library needs to be installed.
The command to run the code is-
 python3 Q4b.py