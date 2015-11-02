function checkGradients( lambda )
%CHECKGRADIENTS Creates a small neural network to check the
%backpropagation gradients
%   CHECKGRADIENTS( lambda ) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by the backprop code and the numerical gradients (computed
%   using computeNumericalGradient), which should result in very similar values.
%

if ~exist( 'lambda', 'var' ) || isempty( lambda )
    lambda = 0;
end

inputLayerSize = 784;
hiddenLayerSize = 25;
numberOfLabels = 10;
numTrainingCases = 5;

% Generate some test data
firstLayerWeights = initDebugWeights( hiddenLayerSize, inputLayerSize );
secondLayerWeights = initDebugWeights( numberOfLabels, hiddenLayerSize );
% Reusing debugInitializeWeights to generate X
X  = initDebugWeights( numTrainingCases, inputLayerSize - 1 );
y  = 1 + mod( 1 : numTrainingCases, numberOfLabels )';

% Unroll parameters
weightsAsVector = [ firstLayerWeights( : ); secondLayerWeights( : ) ];

% Short hand for cost function
costFunc = @( p ) neuralNetCostFunc( p, inputLayerSize, hiddenLayerSize, ...
                               numberOfLabels, X, y, lambda );

[ ~, gradient ] = costFunc( weightsAsVector );
numericalGradient = computeNumericalGradient( costFunc, weightsAsVector );

% Display gradient computation for visial examination
disp( [ numericalGradient gradient ] );

% Evaluate the norm of the difference between two solutions.  
difference = norm( numericalGradient - gradient ) / norm( numericalGradient + gradient );

fprintf( 'Relative Difference: %g\n', difference );

end
