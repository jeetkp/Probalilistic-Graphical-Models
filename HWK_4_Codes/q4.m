clear all ; close all; clc;
%Given 
xP=[1 0 1 1 1 0 1 1; % Politics
0 0 0 1 0 0 1 1;
1 0 0 1 1 0 1 0;
0 1 0 0 1 1 0 1;
0 0 0 1 1 0 1 1;
0 0 0 1 1 0 0 1];
%Given
xS=[1 1 0 0 0 0 0 0; % Sport
0 0 1 0 0 0 0 0;
1 1 0 1 0 0 0 0;
1 1 0 1 0 0 0 1;
1 1 0 1 1 0 0 0;
0 0 0 1 0 1 0 0;
1 1 1 1 1 0 1 0];
SizeP =  size(xP,1);
SizeS =  size(xS,1);
% Prior Prob of politics 
PoliticsProb = SizeP/(SizeP +SizeS); 
%Prior Prob of Sports
SportsProb =1-PoliticsProb;

%Averages of all the given datasets
mP = mean(xP);
mS = mean(xS);

xd=[1 0 0 1 1 1 1 0]; % test point

%Probability Politics wrt test set
PoliticsGivenTest = PoliticsProb*prod(mP.^xd.*(1-mP).^(1-xd)); 
%Probability Sports wrt test set
SportsGivenTest = SportsProb*prod(mS.^xd.*(1-mS).^(1-xd)); 

%Both dont add to up to 1. So we normailize the prob of sports and Politics
ProbOfPolitics = PoliticsGivenTest/(PoliticsGivenTest+SportsGivenTest);
disp('Input Test Data ');
xd
disp('Probability of Input test data being about politics is ');
ProbOfPolitics