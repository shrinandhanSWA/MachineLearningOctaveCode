function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

this_h = sigmoid(X * theta); %the hyothesis function

% Calculating the cost
b1 = transpose(1-y) * log(1 - this_h);
b2 = transpose(y) * log(this_h);
a = b1 + b2;
J = (- (1/m)) * a;

% Regularizing the cost
b = theta .^2;
b(1) = theta(1);
b = b .* (lambda/(2*m));
J = J + sum(b) - b(1);

% Calculating the grad
grad = zeros(size(theta));
grad = transpose(X) * (sigmoid(X * theta) - y);

%Regularizing the grad
grad = grad .* (1/m);
old_val = grad(1);
grad = grad + (lambda/m)*theta;
grad(1) = old_val;

end
