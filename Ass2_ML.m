%% Initializations
clc;
clear all;
close all;

%% Load the data
ds = tabularTextDatastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
X = T{:,1:13};
y = T{:,14};

%% plotData
posHeartDis = find(y==1);
negHeartDis = find(y==0);

plot(X(posHeartDis,1),X(posHeartDis,2),'kx','MarkerSize',5);
hold on

plot(X(negHeartDis,1),X(negHeartDis,2),'ko','MarkerSize',5,'color','r');

legend('Has Heart Disease','Doesnt have Heart Disease');

%% Compute cost and gradient
[m,n] = size(X);
initialTheta = zeros((n+1),1);
[J, grad] = computeCost(initialTheta, X, y);

%% run optimization algorithm
options = optimset('gradObj','on','MaxIter',400);
theta = fminunc(@(t)computeCost(t,X,y),initialTheta,options);

%% Check acuuracy
W = [ones(m,1) X];
p = round(sigmoid(W*theta));

accuracy = mean(double(p == y)*100)
