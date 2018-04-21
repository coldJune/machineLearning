function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% 增加一个偏置单位
a_1 = [ones(m,1) X];
% 第一层到第二层
z_2 = sigmoid(a_1*Theta1');
% 增加一个偏置单位
a_2 =[ones(size(z_2)(1),1) z_2];
% 第二层到输出层
z_3 = sigmoid(a_2*Theta2');
[value p] = max(z_3,[],2);






% =========================================================================


end
