function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

% Generate array for C and Sigma we wish to test
values = [ 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
valuesC = values;
valuesS = values;

maxErr = 0;
bestS = 0;
bestC = 0;

% Iterate C
for testC = valuesC
    % Iterate Sigma
    for testS = valuesS
        
        % Train that values on test set with the above values
        model = svmTrain(X, y, testC, @(x1, x2) gaussianKernel(x1, x2, testS));
        
        % Test the model on CV Set (Here return as 200x1 mat)
        pred = svmPredict(model, Xval)
        % Get pred error, the higher the value, the lesser errors there are
        predError = mean(double(pred == yval));
        
        % Check which has the best error
        if predError > maxErr
            maxErr = predError;
            C = testC;
            sigma = testS;
        end;
        
    end;
end;





% =========================================================================

end
