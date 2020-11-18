function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

a = 0;

% Calculating the cost
for tries = 1:m
this_h = sigmoid(X * theta); %the hyothesis function
  b1 = (1-y(tries)) * log(1 - this_h(tries));
  b2 = y(tries) * log(this_h(tries));
  a = a + b1 + b2;
J = - (1/m) * a;

% Calculating the gradients
grad = zeros(size(theta));

for tries = 1:m

 temp = [X(tries, 1); X(tries, 2); X(tries, 3)];
 grad = grad + ((sigmoid(X*theta))(tries, 1) - y(tries)) * temp;

end

grad = grad .* (1/m);

end
