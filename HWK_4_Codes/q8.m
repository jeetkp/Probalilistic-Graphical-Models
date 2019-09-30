clc; clear all; close all;
import brml.*
check = load('sequences.mat');
NoVisibleStates = 4; %'A''C''G''T'
NoClusters = 2;
for i=1:20 %Changing A to 1, C to 2 , G to 3 , T to 4
    %Check1 being generated with changed values
	check1{i}(findstr(check.sequences{i},'A'))= 1;
	check1{i}(findstr(check.sequences{i},'C'))=2;
	check1{i}(findstr(check.sequences{i},'G'))=3;
	check1{i}(findstr(check.sequences{i},'T'))=4;
end
variable.maxit=50;
variable.plotprogress=1;
%Picked from mixMarkov code
[ph,pv1gh,pvgvh,loglikelihood,phgv]=mixMarkov(check1,NoVisibleStates,NoClusters,variable);
for i=1:20
    disp("Sequence is:  ");
    disp(check.sequences{i});
	if phgv{i}(1)>0.5
        disp("Sequence belongs to Group 2 ");
    else
        disp("Sequence belongs to Group 1 ");
	end
end
disp('Log Likelihood is ');
loglikelihood