clear all; close all; clc;
import brml.*
%Using BRML library Documentation
x= load('EMprinter.mat');
x=x.x;
p{1} = array(1,condp(rand(1,2))); %Fuse
p{2} = array(2,condp(rand(1,2))); % Drum
p{3} = array(3,condp(rand(1,2))); % TOner
p{4} = array(4,condp(rand(1,2))); % Paper
p{5} = array(5,condp(rand(1,2))); %Roller
p{6} = array([6 1],condp(rand(2,2),1)); %burning
p{7} = array([7 2 3 4],condp(rand(2,2,2,2),1)); % Quality 
p{8} = array([8 1 4],condp(rand(2,2,2),1)); % Wrinkled 
p{9} = array([9 4 5],condp(rand(2,2,2),1)); % Multiple Pages
p{10} = array([10 1 5],condp(rand(2,2,2),1)); % Paper Jam

check.tol=0.0001; %Tolerance
check.maxiterations=30;  %Number of Iterations
[p loglikelihood]=EMbeliefnet(p,x,check);

jointprob = multpots(p); %Using inbuilt Function
jointprob1 = p{1}; % Finding it ourselfs.
for j = 2:10
    jointprob1 = jointprob1* p{j};
end
disp('Probability of a drum unit problem given the evidence is:')
disp('First value for No and second value for yes  using Joint probability found by me')
% No == 1 % Yes ==2
%Given conditions No Burning smell==6 so we set to 1
%Given conditions No wrinkled paper==8 so we set to 1
%Given conditions Poor Paper quality==7 so we set to 2
%Prob of drum needs to be found which is 2 as per above declaration
% Joint Prob Multiplication 
disptable(condpot(setpot(jointprob1,[8 6 7], [1 1 2]),2),'2');
% Joint Prob using Multpots
disp('First value for No and second value for yes  using multpot function')
disptable(condpot(setpot(jointprob,[8 6 7], [1 1 2]),2),'2');
