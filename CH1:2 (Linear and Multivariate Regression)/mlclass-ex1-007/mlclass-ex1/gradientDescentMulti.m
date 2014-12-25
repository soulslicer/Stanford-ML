function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
features = size(X , 2);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % The number of features is the theta size
    % Setup a matrix for the summation matrix
        % If features = 3, then we have [0;0;0]
        
    summationmatrix = zeros(features,1);
    for i=1:features
        summationmatrix(i,:) = sum(((X * theta) - y).*X(:,i))
        % Iterate through each theta value to solve
    end
        
    theta = theta - ((1/m)*alpha*summationmatrix)








    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
