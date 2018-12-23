function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

prod = zeros(m,1);
for j=1:m
   prod(j) = X(j,:)*theta; 
end

prod2 = sigmoid(prod);

for i=1:m
    if y(i)==1
        J+=(log(prod2(i)));
    else
        J+=(log(1-prod2(i)));
    end
end
J/=(-1*m);

theta_length = length(theta);
for k=1:theta_length
    fact = 0;
    for l=1:m
        fact+=((prod2(l)-y(l))*(X(l,k)));
    end
    grad(k) = (fact)/m;
end

% =============================================================

end
