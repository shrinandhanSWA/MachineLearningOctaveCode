function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
m = size(X, 2); %calculating the no of features

for feature = 1:m
    mu(feature, 1) = mean(X_norm(:, feature)); %mean of this feature
    sigma(feature, 1) = std(X_norm(:, feature)); %standard deviation of this feature
    X_norm(:, feature) = X_norm(:, feature) - mu(feature, 1);
    X_norm(:, feature) = X_norm(:, feature) ./ sigma(feature, 1);

 end

end
