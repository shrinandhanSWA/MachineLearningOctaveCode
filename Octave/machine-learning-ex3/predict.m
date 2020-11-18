function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% p is what will be returned
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
z2 = X * transpose(Theta1);
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * transpose(Theta2);
a3 = sigmoid(z3);
[x, p] = max(a3, [], 2); 
%extra step needed because it is a one vs all implementation i.e. which probability is the highest

end
