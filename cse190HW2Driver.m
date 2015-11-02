% Read in data
[ images, labels ] = read_mnist_training_files( '~/Downloads/train-images-idx3-ubyte', ...
    '~/Downloads/train-labels-idx1-ubyte', 60000, 0 );

% Initialize variables
numTrainingCases = 50000;
numTestCases = 10000;
numberOfFeatures = 784;
hiddenLayerSize = 25;
outputLayerSize = 10;
numberOfEpochs = 5;
regularizationTerm = 0;
learningRate = 0.1;
testSetAccuracy = zeros( numberOfEpochs, 1 );
trainingSetAccuracy = zeros( numberOfEpochs, 1 );
momentum = 0.0;
previousGradient1 = 0;
previousGradient2 = 0;

% Map 0's to 10's for easier indexing later
labels( labels == 0 ) = 10;

% Transform single vector of data into a matrix representation of the data
trainingCases = zeros( numTrainingCases, numberOfFeatures );
testCases = zeros( numTestCases, numberOfFeatures );
for i = 1 : numTrainingCases
    trainingCases( i, : ) = reshape( images( :, :, i ).', numberOfFeatures, 1);
end

for i = 1 : numTestCases
    testCases( i, : ) = reshape( images( :, :, i + numTrainingCases ).', ...
        numberOfFeatures, 1);
end

firstLayerWeights = randomlyInitWeights( numberOfFeatures, hiddenLayerSize );
secondLayerWeights = randomlyInitWeights( hiddenLayerSize, outputLayerSize );
weightsAsVector = [ firstLayerWeights( : ); secondLayerWeights( : ) ];

% Train neural net using stochstic gradient descent
for i = 1 : numberOfEpochs
    % Randomize the order in which training cases are processed
    testCaseIndices = randperm( numTrainingCases );
    
    for j = 1 : numTrainingCases
        index = testCaseIndices( 1, j );
        
        [ cost, gradient ] = neuralNetCostFunc( weightsAsVector, ...
            numberOfFeatures, hiddenLayerSize, outputLayerSize, ... 
            trainingCases( index, : ), labels( index ), regularizationTerm );

        firstLayerGradient = reshape( gradient( 1 : hiddenLayerSize * ( numberOfFeatures ...
            + 1 ) ), hiddenLayerSize, ( numberOfFeatures + 1 ) );
        secondLayerGradient = reshape( gradient( ( 1 + ( hiddenLayerSize * ( numberOfFeatures ...
            + 1 ) ) ) : end ), outputLayerSize, ( hiddenLayerSize + 1 ) );

        firstLayerWeights = firstLayerWeights - learningRate * firstLayerGradient ...
            + momentum * previousGradient1;
        secondLayerWeights = secondLayerWeights - learningRate * secondLayerGradient ...
            + momentum * previousGradient2;

        previousGradient1 = firstLayerGradient;
        previousGradient2 = secondLayerGradient; 
        
        weightsAsVector = [ firstLayerWeights( : ); secondLayerWeights( : ) ];
        
    end
    
    % Test neural net accuracy per gradient descent update
    predictions = predict( firstLayerWeights, secondLayerWeights, testCases );
    testSetAccuracy( i ) = mean( double( predictions == labels( numTrainingCases + 1 : ...
        numTestCases + numTrainingCases ) ) * 100 );

    predictions = predict( firstLayerWeights, secondLayerWeights, trainingCases );
    trainingSetAccuracy( i ) = mean( double( predictions == labels( 1 : numTrainingCases ) ) ...
        * 100 );
end
