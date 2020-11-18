function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

n = size(X, 2); %number of features
m = length(y); %number of training examples
theta = zeros(n, 1); %initialize theta

X_transposed = transpose(X); %need to transpose X now

big_X = X_transposed * X;

small_X = pinv(big_X);

theta_temp = small_X * X_transposed;

theta = theta_temp * y;

end
