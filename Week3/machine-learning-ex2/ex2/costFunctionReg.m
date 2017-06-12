function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

n = length(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% calculate h vector (m x 1)
h = sigmoid(X*theta);

% non-bias theta values
new_theta = theta(2:n,:);

% regularized J value
J = (-y.'*log(h) - (1-y.')*log(1-h))/m + lambda/(2*m)*(new_theta.')*new_theta;


% vectorized gradient without regularization
grad = ((h-y).'*X)/m;

% added regularization
grad(2:n) = grad(2:n) + lambda/m*new_theta.';




% =============================================================

end
