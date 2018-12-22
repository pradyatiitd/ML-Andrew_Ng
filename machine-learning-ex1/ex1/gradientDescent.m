function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


cost1 = 0
for i=1:m
    cost1+=(theta(1) + (theta(2)*X(i,2)) - y(i));
end
cost1*=alpha
cost1/=m

cost_2 = 0
for j=1:m
    cost_2+=((theta(1) + (theta(2)*X(j,2)) - y(j))*X(j,2));
end
cost_2*=alpha
cost_2/=m

theta(1)-=cost1
theta(2)-=cost_2




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
