function J = costFunctionJ(X, y, theta)

m = size(X, 1); %training set size
predictions = X*theta; %predictions matrix via matrix multiplication
sqrErrors = (predictions-y).^2; %sqare erros by subtracting the 2 matrices then squaring each element

J = 1/(2*m) * sum(sqrErrors); %summing the matrix and multiplying by 1/2*m