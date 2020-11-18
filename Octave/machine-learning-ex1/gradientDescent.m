function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    a = zeros(2, 1);

    for tries = 1:m
        temp = [X(tries, 1); X(tries, 2)];
        a = a + ((X*theta)(tries, 1) - y(tries)) * temp;
    end

    theta = theta - alpha * (1/m) * a;
 
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
 end
end