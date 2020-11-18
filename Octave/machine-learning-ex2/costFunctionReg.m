function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2); %number of features

a = 0; b=0;

% Calculating the normal cost
for tries = 1:m
this_h = sigmoid(X * theta); %the hyothesis function loll
  b1 = (1-y(tries)) * log(1 - this_h(tries));
  b2 = y(tries) * log(this_h(tries));
  a = a + b1 + b2;
J = - (1/m) * a;

% Normalizing the cost
for tries = 2:n 
    b = b + theta(tries)^2;
end
b = (b * lambda) / (2*m);
J = J + b;

% Calculating the gradient
grad = zeros(size(theta));
for tries = 1:m
 temp = X(tries, 1);
 for tries2 = 2:size(theta)
    temp = [temp; X(tries, tries2)];
end
 grad = grad + ((sigmoid(X*theta))(tries, 1) - y(tries)) * temp;

end

% Normalizing the grad
grad = grad .* (1/m);
old_val = grad(1);
grad = grad + (lambda/m)*theta;
grad(1) = old_val;