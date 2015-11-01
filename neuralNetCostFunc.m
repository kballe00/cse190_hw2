function [ J, gradient ] = neuralNetCostFunc(weights, ...
                                   inputLayerSize, ...
                                   hiddenLayerSize, ...
                                   outputLayerSize, ...
                                   X, y, lambda)
%NEURALNETCOSTFUNC Implements the neural network cost function for a two layer
%neural network which performs classification using softmax.
%   [J gradient] = NEURALNETCOSTFUNC(weights, hiddenLayerSize, outputLayerSize, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   weights and need to be converted back into the weight matrices. 
% 
%   The returned parameter gradient should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape weights back into weight matrices
firstLayerWeights = reshape( weights( 1 : hiddenLayerSize * ( inputLayerSize ...
    + 1 ) ), hiddenLayerSize, ( inputLayerSize + 1 ) );

secondLayerWeights = reshape( weights( ( 1 + ( hiddenLayerSize * ( inputLayerSize ...
    + 1 ) ) ) : end ), outputLayerSize, ( hiddenLayerSize + 1 ) );

% Intialize variables
numTrainingExamples = size( X, 1 );

J = 0;

% Add bias to input
X = [ ones( numTrainingExamples, 1 ) X ];

bigDelta2 = zeros( outputLayerSize, hiddenLayerSize + 1);
bigDelta1 = zeros( hiddenLayerSize, inputLayerSize + 1 );

% Iterate over every training example, and compute the error
for i = 1 : numTrainingExamples
    layerOneOutput = X( i, : ).';
    layerTwoInput = firstLayerWeights * X( i, : ).';
    layerTwoOutput =  [ 1; sigmoid( layerTwoInput ) ];
    classifierUnitInput = ( secondLayerWeights * layerTwoOutput );
    
    % Create vector to represent what the output should be
    expectedOutput = zeros( outputLayerSize, 1 );
    % Set the node that should have the highest probability to 1
    correctAnswer = y( i );
    expectedOutput( correctAnswer ) = 1;    

    prediction = softmaxActivation( classifierUnitInput.' ).';
    
    % Calculate the cost/error
    J = J - sum( expectedOutput .* log( prediction ) - ( 1 -  expectedOutput ...
        ) .* log( 1 - prediction ) );   
        
    littleDelta3 = prediction - expectedOutput;
    
    % Compute error for the second layer, based on the output layer error
    littleDelta2 = ( secondLayerWeights.' * littleDelta3 ) .* ...
        layerTwoOutput .* ( 1 - layerTwoOutput );
  
    % Accumulate error over training examples
    bigDelta2 = bigDelta2 + littleDelta3 * layerTwoOutput.';
    bigDelta1 = bigDelta1 + littleDelta2( 2 : end ) * layerOneOutput.';
end

% Overall cost needs to be divided by number of training examples
J = J / numTrainingExamples;

tempFLW = firstLayerWeights( :, 2 : end );
tempSLW = secondLayerWeights( :, 2 : end );

% Normalize J
J = J + sum( ( lambda / ( 2 * numTrainingExamples ) ) * ( tempFLW(:) ...
    .^ 2 ) ) + sum( ( lambda / ( 2 * numTrainingExamples ) ) * ...
    ( tempSLW(:) .^ 2 ) );

% Overall gradients needs to be divided by number of training examples, and
% regularization also needs to be added (skip the first column, since it 
% represents the bias)
gradientFLW = ( bigDelta1 + lambda * [ zeros( size( bigDelta1, 1 ), 1 ) ...
    firstLayerWeights( :, 2 : end ) ] ) / numTrainingExamples;
gradientSLW = ( bigDelta2 + lambda * [ zeros( size( bigDelta2, 1 ), 1 ) ...
    secondLayerWeights( :, 2 : end ) ] ) / numTrainingExamples;

% Unroll gradients
gradient = [ gradientFLW( : ); gradientSLW( : ) ];
end
