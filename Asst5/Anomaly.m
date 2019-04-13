%% Clears
clc
clear all
close all

%% Input Data
ds = tabularTextDatastore('house prices data training data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
X = T{:,4:21};
Y=T{:,3};

%% Gaussian
[m, n] = size(X);
mu = ((1/m)*sum(X))';
sigma2 = ((1/m)*sum((X-mu').^2))';

Xval = X(1:100,:);
Yval = X(1:100,:);

%% Multivariate Gaussian
p = multivariateGaussian(X, mu, sigma2);
Pval = multivariateGaussian(Xval, mu, sigma2);

%% Evaluate Epsilon
[epsilon F1] = selectThreshold(Yval, Pval);
outliers = find(Pval < epsilon);
