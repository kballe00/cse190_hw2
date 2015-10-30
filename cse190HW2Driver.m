% Read in data
[ images, labels ] = read_mnist_training_files( '~/Downloads/train-images-idx3-ubyte', ...
    '~/Downloads/train-labels-idx1-ubyte', 20000, 0 );

% Initialize variables
numTrainingCases = 20000;
numberOfFeatures = 784;
hiddenLayerSize = 25;
outputLayerSize = 10;
numberOfEpochs = 50;
regularizationTerm = 100;
learningRate = 1;

% Map 0's to 10's for easier indexing later
labels( labels == 0 ) = 10;

% Transform single vector of data into a matrix representation of the data
trainingCases = zeros( numTrainingCases, numberOfFeatures );
for i = 1 : numTrainingCases
   trainingCases( i, : ) = reshape( images( :, :, i ).', numberOfFeatures, 1);
end

firstLayerWeights = randomlyInitWeights( numberOfFeatures, hiddenLayerSize );
secondLayerWeights = randomlyInitWeights( hiddenLayerSize, outputLayerSize );
weightsAsVector = [ firstLayerWeights( : ); secondLayerWeights( : ) ];

% Train neural net using batch processing
for i = 1 : numberOfEpochs
    [ cost, gradient ] = neuralNetCostFunc( weightsAsVector, ...
        numberOfFeatures, hiddenLayerSize, outputLayerSize, trainingCases, ...
        labels, regularizationTerm );
    
    firstLayerGradient = reshape( gradient( 1 : hiddenLayerSize * ( numberOfFeatures ...
        + 1 ) ), hiddenLayerSize, ( numberOfFeatures + 1 ) );
    secondLayerGradient = reshape( gradient( ( 1 + ( hiddenLayerSize * ( numberOfFeatures ...
        + 1 ) ) ) : end ), outputLayerSize, ( hiddenLayerSize + 1 ) );
    
    firstLayerWeights = firstLayerWeights - learningRate * firstLayerGradient;
    secondLayerWeights = secondLayerWeights - learningRate * secondLayerGradient;
    
    weightsAsVector = [ firstLayerWeights( : ); secondLayerWeights( : ) ];
end

% Test neural net accuracy
predictions = predict( firstLayerWeights, secondLayerWeights, trainingCases );
fprintf('\nTraining set accuracy: %f\n', mean( double( predictions == labels ) ) ...
    * 100 );