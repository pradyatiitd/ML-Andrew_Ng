function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

J2 = 0;
theta_length = length(theta);
for j=2:theta_length
    J2+=(theta(j)*theta(j));
end
J2*=(lambda/(2*m));
J+=J2;


theta_length = length(theta);
for k=1:theta_length
    fact = 0;
    for l=1:m
        fact+=((prod2(l)-y(l))*(X(l,k)));
    end
    grad(k) = (fact)/m;
    if k!=1
        grad(k)+=((lambda/m)*(theta(k)));
    end
end



% =============================================================

end
