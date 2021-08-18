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
sigma = 0.3;

C_vect =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vect =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% Values from ex6.m
x1 = [1 2 1];
x2 = [0 4 -1];

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

for i = 1:length(C_vect)
    for j = 1:length(sigma_vect)
        % From ex6.m line 108
        model= svmTrain(X, y, C_vect(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(j)));
        pred = svmPredict(model,Xval);
        acc_matrix(i,j) = mean(double(pred ~= yval));
    end
end

% Find coordinates of minimum predicted error
[row, col] = find(acc_matrix == min(acc_matrix(:)), 1);

C = C_vect(row);
sigma = sigma_vect(col);

% =========================================================================

end
