function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_arr = [0.1;0.3;1;3;0.01;0.03];
sigma_arr = [0.01;0.03;0.1;0.3;1;3];
C_opt = 0;
sigma_opt = 0;
curr_min = 100000;
m = size(C_arr,1);
n = size(sigma_arr);
for i=1:m
    for j=1:n
        6*i+j
        model = svmTrain(X,y,C_arr(i),@(x1,x2) gaussianKernel(x1,x2,sigma_arr(j)));
        predictions = svmPredict(model,Xval);
        curr = mean(double(predictions ~= yval));
        if curr<curr_min
            curr_min = curr;
            C_opt = C_arr(i);
            sigma_opt = sigma_arr(j);
        end
    end
end

C = C_opt;
sigma = sigma_opt;

% =========================================================================

end
