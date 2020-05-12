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

C_seq=[0.01,0.03,0.1,0.3,1,3,10,30];
sigma_seq=C_seq;

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

predict_matrix=zeros(size(C_seq,2),size(sigma_seq,2));
for i=1:size(C_seq,2)
    for j=1:size(sigma_seq,2)
        model= svmTrain(X, y, C_seq(i), @(x1, x2) gaussianKernel(x1, x2, sigma_seq(1,j)));
        predictions = svmPredict(model, Xval);
        error =mean(double(predictions ~= yval));
        predict_matrix(i,j)=error;
    end
end


[x y]=find(predict_matrix==min(min(predict_matrix)));

C=C_seq(x(1));
sigma=sigma_seq(x(1));


disp(C)
disp(sigma)

% =========================================================================

end

