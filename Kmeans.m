%% Clears
clc
clear all
close all

%% Input Data
ds = tabularTextDatastore('house prices data training data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
X = T{:,4:21};
[n,m]= size(X);

%% Kmeans method 
%%% IDX is cluster membership for each datapoint in vector X
%%% SUMD is the sum of point-to-cluster-centroid distances.
[IDX,centroidLoc,SUMD,clustersNumber]=kmeans_opt(X,m);

%% PCA Starting
fprintf('Running PCA ...\n');

%% Corr, Cov and svd
Corr_x = corr(X);  
x_cov=cov(X) ; 
[U, S, V] = svd(x_cov);
[n,m]= size(X);

%% Eigen Values
eigen_vals = diag(S);
eigen_vals=eigen_vals';

%% Compute K
alpha = 1-(sum(eigen_vals(1:0))/sum(eigen_vals(1:m)));
for K=1:m
    if alpha>0.001
    alpha = 1-(sum(eigen_vals(1:K))/sum(eigen_vals(1:m)));
    end
    if alpha<=0.001
        break
    end
end

%% Reduced Data R
  U_reduce = U(:,[1:K])';   % K x n
  R = U_reduce * X';        % K x m
  R=R';
  
%% Kmeans method on reduced data
%%% IDX is cluster membership for each datapoint in vector X
%%% SUMD is the sum of point-to-cluster-centroid distances.
[n,m]= size(R);
[IDXReduced,centroidLocReduced,SUMDReduced,clustersNumberReduced]=kmeans_opt(R,m);