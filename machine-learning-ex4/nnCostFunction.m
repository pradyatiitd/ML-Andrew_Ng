function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

dummy_X = X;
X = [ones(m,1) X];
input_matrix = X*Theta1';
input_matrix = sigmoid(input_matrix);
input_matrix = [ones(m,1) input_matrix];
input_matrix = input_matrix*Theta2';
input_matrix = sigmoid(input_matrix);

for i=1:m
    for j=1:num_labels
        if y(i)==j
            J-=(log(input_matrix(i,j)));
        else
            J-=(log(1-input_matrix(i,j)));
        end
    end
end
J/=m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for idx=1:m
    da_1 = dummy_X(idx,:)';
    % size(da_1) = 400 1
    a_1 = [1;da_1];
    z_2 = Theta1*a_1;
    % size(z_2) = 25 1
    da_2 = sigmoid(z_2);
    a_2 = [1;da_2];
    z_3 = Theta2*a_2;
    % size(z_3) = 10 1
    da_3 = sigmoid(z_3);
    out_mat = zeros(num_labels,1);
    out_mat(y(idx)) = 1;
    del_3 = da_3 - out_mat;
    del_2 = 0;
    del_2 = (Theta2'*del_3);
    del_2(1,:) = [];
    del_2 = del_2.*sigmoidGradient(z_2);
    grad1 = del_2 * a_1';
    Theta1_grad += grad1;
    grad2 = del_3*a_2';
    Theta2_grad+=grad2;
end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

mat1 = Theta1.*Theta1;
mat2 = Theta2.*Theta2;
sum1 = sum(mat1,1);
sum1 = sum(sum1,2) - sum1(1);
sum2 = sum(mat2,1);
sum2 = sum(sum2,2) - sum2(1);
sum1+=sum2;
sum1*=lambda;
sum1/=(2*m);
J+=sum1;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

mat1 = zeros(size(Theta1));
m1 = size(Theta1,1);
m2 = size(Theta1,2);
for i=1:m1
    for j=2:m2
        mat1(i,j) = lambda*Theta1(i,j);
    end
end

mat2 = zeros(size(Theta2));
m1 = size(Theta2,1);
m2 = size(Theta2,2);
for i=1:m1
    for j=2:m2
        mat2(i,j) = lambda*Theta2(i,j);
    end
end

Theta1_grad+=mat1;
Theta2_grad+=mat2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad/=m;

end
