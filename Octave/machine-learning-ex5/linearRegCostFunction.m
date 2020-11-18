function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
this_h = X*theta; %hypothesis function, shud be the same size as y 
this_h = this_h - y;
this_h = this_h .^2;
J = sum(this_h);
J = J * (1/(2*m));

temp_theta = theta .^2;
temp_theta = temp_theta(2:end);
sum = sum(temp_theta);
J = J + sum*(lambda/(2*m));

grad = zeros(size(theta));


for tries = 1:m,
temp = [X(tries, 1)];
    for tempyyy = 2 : size(grad),
        temp = [temp; X(tries,tempyyy)];
    end;
    grad = grad + (((X*theta)(tries, 1) - y(tries)) * temp);
end;

grad = grad * (1/m);

temp = grad(1); 
grad = grad + theta*(lambda/m);
grad(1) = temp;

grad = grad(:);

end
