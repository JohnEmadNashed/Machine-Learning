%% Clears
clc
clear all
close all

%% Input Data
ds = tabularTextDatastore('house prices data training data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
T = read(ds);
X = T{:,4:21};

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
  
%% Approximate Data A.
  U_reduce = U(:,1:K);   % n x K
  A = R' * U_reduce'; % m x n
  
%% Error
R=R';
for i=1:m
Error = 1/m*sum(A(i)-R(i));
end

%% Gradient Descent and Logistic Regresion Ana
fprintf('Running gradient descent ...\n');
 
% Xori=X;
% [mOri,nOri]=size(X);
 
X=R;
[m,n]=size(X);
 
Y=T{:,3}/mean(T{:,3});
Alpha=.01;
Theta=zeros(n,1);
 
k=1;
E(k)=(1/(2*m))*sum((R*Theta-Y).^2); %Initial error
count=1;
while count==1
Alpha=Alpha*1; %law 7abeb a8yr el alpha fe kol iteration mmkn a8yr el factor 1
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
k=k+1;
E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
if E(k-1)-E(k)<0
    break  %law el error byzed by2fl mmkn ab2a arwsh wabd2 a8yr parameters 3shan a2ll el error
end
q=(E(k-1)-E(k))./E(k-1);
if q <.001; %percentage decrease of error
    count=0;
end
end
 
%% Plot the convergence graph of Linear Regression
figure(1);
plot(1:numel(E), E, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J'); 
