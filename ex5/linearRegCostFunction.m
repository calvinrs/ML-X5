function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%% Adapted from ML-X3
%calc the hypothesis function for all X;

%Xbias = [ones(size(X,1), 1) X];

%linear hypothesis - theta(0) + x*theta(1)
H = X * theta;

D = H - y; %diff versus observed
S = D' * D; % an array transposed times the array gives the sum of squares

%regularisation term
unbiasedTheta = theta(2:size(theta,1),:);
R = (lambda/(2*m)) * (unbiasedTheta' * unbiasedTheta);

J = S/(2*m) + R;



% GRADIENTS

thetaJtoM =  [0; unbiasedTheta]; % as theta(0) = 0; don't regularise bias term
%gradients + regularisation term (0 for first coeff.)
grad = ((1 /m) * (X' * D)) + ((lambda/m) * thetaJtoM);


% =========================================================================

grad = grad(:);

end
