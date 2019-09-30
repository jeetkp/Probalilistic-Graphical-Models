In this code the basic network is being defined with all the data given of estimated and true probabilities. 

This has a bayesian network graph, which does all initialization of all data relating to probabilities in the nodes that we defined and also queries and samples them as well. 

There is a class called node_element which is basically a structure of the node. 

This code basically given out the joint probabilites as per the order of the data given to it. 
As of now the input is hardcoded to [1,1,1,1,1,1,1,1,1,1,1,0]
This can be changed to any list of 12 binary numbers and the joint distribution will be obtained. 

The code is very efficient and runs within seconds. 

The library requirements are numpy(1.16.2)(in my system) 

The command to run the code is-
 python3 Q4.py