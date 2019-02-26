%% Clearances
clc
clear all
close all

%% CSV File Read
% house_data_complete.csv
% house_prices_data_training_data.csv
ds = tabularTextDatastore('house_data_complete.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',25000);
ds.SelectedVariableNames = {'price','bedrooms','bathrooms','sqft_living','sqft_lot','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','lat', 'long','sqft_living15','sqft_lot15'};

T = read(ds);

% X = T{:,4:21};
y = T{:,1}; 
m = length(y);

%% Plots
% figure(2)
% stem (T{:,2},T{:,1},':x') %Hyp = X^2
% xlabel('Number of Bedrooms');
% ylabel('Price of House');
% 
% figure(3)
% stem (T{:,3},T{:,1},':x') %Hyp = X
% xlabel('Number of Bathrooms');
% ylabel('Price of House');
% 
% figure(4)
% stem (T{:,4},T{:,1},':x') %Hyp = X
% xlabel('SqFT of Living room');
% ylabel('Price of House');
% 
% figure(5)
% stem (T{:,5},T{:,1},':x') %
% xlabel('SqFT Total');
% ylabel('Price of House');
% 
% 
% figure(6)
% stem (T{:,6},T{:,1},':x') %
% xlabel('Waterfront');
% ylabel('Price of House');
% 
% figure(7)
% stem (T{:,7},T{:,1},':x') %
% xlabel('Number of views');
% ylabel('Price of House');
% 
% figure(8)
% stem (T{:,8},T{:,1},':x') %Hyp = X^2
% xlabel('condition');
% ylabel('Price of House');
% 
% figure(9)
% stem (T{:,9},T{:,1},':x') %Hyp = Exp(X)
% xlabel('Grade');
% ylabel('Price of House');
% 
% figure(10)
% stem (T{:,10},T{:,1},':x') %Hyp = X
% xlabel('SQFT above');
% ylabel('Price of House');
% 
% figure(11)
% stem (T{:,11},T{:,1},':x') %Hyp = X
% xlabel('SQFT basement');
% ylabel('Price of House');
% 
% figure(12) %Exclude Years 0
% stem (T{:,12},T{:,1},':x') %
% xlabel('Yr Built');
% ylabel('Price of House');
% 
% figure(13)
% stem (T{:,13},T{:,1},':x') %
% xlabel('Yr renovated');
% ylabel('Price of House');
% 
% figure(14)
% stem (T{:,14},T{:,1},':x') %
% xlabel('Latitude');
% ylabel('Price of House');
% 
% figure(15)
% stem (T{:,15},T{:,1},':x') %
% xlabel('Longitude');
% ylabel('Price of House');
% 
% figure(16) 
% stem (T{:,16},T{:,1},':x') %Hyp = X
% xlabel('SQFT living15');
% ylabel('Price of House');
% 
% figure(17)
% stem (T{:,17},T{:,1},':x') %Hyp = e^-X
% xlabel('SQFT Tot15');
% ylabel('Price of House');

%% Hypotheses

% Hypothesis 1 
% U = T{:,2:15}; % features without price
% U1 = T{:,16:17}; % last 2 features
% U2 = U.^2;
% X =[ones(m,1) U U1 U.^2 U.^3]; %k=254 Iterations
% % U3=log(X);
% % U4=exp(X);
% % X=[ones(m,1) U U1 U.^2 U.^3 U3 U4];

% Hypothesis 2
U = T{:,[2 3 4 5 9 10 11 16 17]}; 
X=[ones(m,1) U]; %k=234 Iterations

% Hypothesis 3
% U1 = T{:,16:17}; % last 2 features 
% X=[ones(m,1) U1]; %k=216 Iterations

% Hypothesis 4
% U = T{:,[3 4 10 11 16]}; % Features with x hyp 3,4,10,11,16
% U_of_Sq = (T{:,[2 8]}.^2); % Features with x^2 hyp figures: 2,8
% U_of_ExpX = exp(T{:,[9]});   % Features with Exp(x) hyp 9
% Uo=[T{:,:}];
% X=[U , U_of_Sq , U_of_ExpX]; %k=73 Iterations

%% Normalization 
fprintf('Normalizing Features ...\n');
n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
end

%% Gradient Descent and Linear Regresion Dr Ashour
% fprintf('Running gradient descent ...\n');
% Y=T{:,1}/mean(T{:,1});
% Alpha=.01;
% Theta=zeros(n,1);
% k=1;
% 
% E(k)=(1/(2*m))*sum((X*Theta-Y).^2); %Initial error
% 
% R=1;
% while R==1
% Alpha=Alpha*1; %law 7abeb a8yr el alpha fe kol iteration mmkn a8yr el factor 1 
% Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
% k=k+1;
% E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
% if E(k-1)-E(k)<0
%     break  %law el error byzed by2fl mmkn ab2a arwsh wabd2 a8yr parameters 3shan a2ll el error
% end 
% q=(E(k-1)-E(k))./E(k-1);
% if q <.001; %percentage decrease of error
%     R=0;
% end
% end

%% Plot the convergence graph of Linear Regression
% figure(1);
% plot(1:numel(E), E, '-b', 'LineWidth', 2);
% xlabel('Number of iterations');
% ylabel('Cost J');

%% Gradient Descent and Logistic Regresion Ana
fprintf('Running gradient descent ...\n');
Y=T{:,1}/mean(T{:,1});
Alpha=.01;
Theta=zeros(n,1);
k=1;

E(k)=(1/(2*m))*sum((X*Theta-Y).^2); %Initial error

R=1;
while R==1
Alpha=Alpha*1; %law 7abeb a8yr el alpha fe kol iteration mmkn a8yr el factor 1 
Theta=Theta-(Alpha/m)*X'*(X*Theta-Y);
k=k+1;
E(k)=(1/(2*m))*sum((X*Theta-Y).^2);
if E(k-1)-E(k)<0
    break  %law el error byzed by2fl mmkn ab2a arwsh wabd2 a8yr parameters 3shan a2ll el error
end 
q=(E(k-1)-E(k))./E(k-1);
if q <.001; %percentage decrease of error
    R=0;
end
end

%% Plot the convergence graph of Linear Regression
figure(1);
plot(1:numel(E), E, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

%% Estimate the price of first house in the dataset (rawData)
% d = T{1,4:21};
% d = (d - mu) ./ sigma;
% d = [ones(1, 1) d];
% price = d * Theta;
% 
% fprintf(['Predicted price of a house ' ...
%          '(using gradient descent):\n $%f\n'], price);
