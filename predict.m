function predictions = predict( firstLayerWeights, secondLayerWeights, ...
    trainingData )
%PREDICT Predict the label of an input given a trained neural network
%   predictions = PREDICT( firstLayerWeights, secondLayerWeights, trainingData )
%   outputs the predicted labels of trainingData given the trained weights
%   firstLayerWeights and secondLayerWeights.

% Initialize variables
numTrainingCases = size( trainingData, 1 );

% Add bias to training data
trainingData = [ ones( numTrainingCases, 1 ) trainingData ];
hiddenLayerOutput = sigmoid( trainingData * firstLayerWeights.' );
% Add bias to hidden layer output
hiddenLayerOutput = [ ones( numTrainingCases, 1 ) hiddenLayerOutput ];
output = softmaxActivation( hiddenLayerOutput * secondLayerWeights.' );

[ ~, predictions ] = max( output, [ ], 2 );
end
