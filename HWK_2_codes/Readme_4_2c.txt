In this code the variable elimination is being done. 
There is a factor class  which saves the factors and its probabilities. Then there is an eliminate function where you pass these nodes and probabilities along with variable to be eliminated to remove the desired node. 

The order of elimination is as of now hard coded from leaves to root. 

This has a bayesian network graph as well, which does all initialization of all data relating to probabilities in the nodes that we defined and also queries and samples them as well. 

There is a class called node_element which is basically a structure of the node. 

All data asked in the question gets prinited. I have attached a screenshot of the data as well in the report. 

The code is very effiecient and runs within seconds. 

The queries are hardcoded in the code. But the code is generalized. So we can change the query variables and observed variables and then variable elimination is performed on it. 

Numpy library needs to be installed.

The command to run the code is-
python3 4.2c\).py