function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X, 2); % net number of features

for iter = 1:num_iters

    a = zeros(n, 1);

    for tries = 1:m

     temp = X(tries, 1);
    for tries_temp = 2:n
        temp = [temp; X(tries, tries_temp);];
     end
    
    a = a + ((X*theta)(tries, 1) - y(tries)) * temp;
    end

    theta = theta - alpha * (1/m) * a;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
