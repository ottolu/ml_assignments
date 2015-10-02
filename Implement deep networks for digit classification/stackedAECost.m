function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

%forward feed
z2 = bsxfun(@plus, stack{1}.w * data, stack{1}.b);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, stack{2}.w * a2, stack{2}.b);
a3 = sigmoid(z3);
z4 = softmaxTheta * a3;
z4 = bsxfun(@minus, z4, max(z4));
a4 = exp(z4);
a4 = bsxfun(@rdivide, a4, sum(a4));

cost = - sum(sum(groundTruth .* log(a4))) ./ m + ...
      lambda * sum(sum(softmaxTheta .^ 2)) / 2;
  
o4 = - (groundTruth - a4);
o3 = (softmaxTheta' * o4) .* dsigmoid(z3);
o2 = (stack{2}.w' * o3) .* dsigmoid(z2);

dw2 = o3 * a2';
db2 = sum(o3, 2);
dw1 = o2 * data';
db1 = sum(o2, 2);

softmaxThetaGrad = (o4 * a3') ./ m + lambda .* softmaxTheta;

stackgrad{1}.w = dw1 ./ m + lambda .* stack{1}.w;
stackgrad{1}.b = db1 ./ m;
stackgrad{2}.w = dw2 ./ m + lambda .* stack{2}.w;
stackgrad{2}.b = db2 ./ m;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function dsigm = dsigmoid(a)
     e_a = exp(-a);
     dsigm = e_a ./ ((1 + e_a).^2); 
end
