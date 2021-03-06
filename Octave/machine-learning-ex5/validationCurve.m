function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = transpose([0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]);
m = length(lambda_vec);
error_train = zeros(m, 1);
error_val = zeros(m, 1);

% You need to return these variables correctly.

for i = 1:m,
    this_lambda = lambda_vec(i);
    this_theta = trainLinearReg(X, y, this_lambda);
    error_train(i) = linearRegCostFunction(X, y, this_theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, this_theta, 0);
end;

end
