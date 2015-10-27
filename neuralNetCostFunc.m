function [J, gradient] = neuralNetCostFunc(weights, ...
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
numberOfFeatures = size( X, 1 );
          
J = 0;

% Add bias to input
X = [ ones( numberOfFeatures, 1 ) X ];

% Holds error for output units
littleDelta3 = zeros( outputLayerSize, 1 );

bigDelta2 = zeros( outputLayerSize, hiddenLayerSize + 1);
bigDelta1 = zeros( hiddenLayerSize, inputLayerSize + 1 );

% Iterate over every feature, and compute error
for i = 1 : numberOfFeatures
    layerOneOutput = X( i, : ).';
    layerTwoInput = firstLayerWeights * X( i, : ).';
    layerTwoOutput =  [ 1; sigmoid( layerTwoInput ) ];
    
    % THIS PART NEEDS TO BE FIXED - IT CURRENTLY USES K-WAY CLASSIFCATION
    % BUT NEEDS TO IMPLEMENT SOFTMAX
    for k = 1 : outputLayerSize
        yForClass = ( y == k );
        classifierUnitInput = layerTwoOutput.' * secondLayerWeights( k, : ).';
        prediction = sigmoid( classifierUnitInput );
        
        J = J - yForClass( i ) * log( prediction ) - ( 1 -  yForClass( i ) ...
            ) * log( 1 - prediction );
        
        littleDelta3( k ) = prediction - yForClass( i );
    end
    
    % Compute error for the second layer, based on the output layer error
    littleDelta2 = ( secondLayerWeights.' * littleDelta3 ) .* ...
        layerTwoOutput .* ( 1 - layerTwoOutput );
    
    % Accumulate error over training examples
    bigDelta2 = bigDelta2 + littleDelta3 * layerTwoOutput.';
    bigDelta1 = bigDelta1 + littleDelta2( 2 : end ) * layerOneOutput.';
end

% Overall cost needs to be divided by number of features
J = J / numberOfFeatures;

tempFLW = firstLayerWeights( :, 2 : end );
tempSLW = secondLayerWeights( :, 2 : end );

% Normalize J
J = J + sum( ( lambda / ( 2 * numberOfFeatures ) ) * ( tempFLW(:) ...
    .^ 2 ) ) + sum( ( lambda / ( 2 * numberOfFeatures ) ) * ...
    ( tempSLW(:) .^ 2 ) );

% Overall gradients needs to be divided by number of features
gradientFLW = bigDelta1 / numberOfFeatures;
gradientSLW = bigDelta2 / numberOfFeatures;

% Unroll gradients
gradient = [ gradientFLW(:) ; gradientSLW(:) ];


end
