function [ probabilityVector ] = softmaxActivation( weights, trainingData )
%softmaxActiviation This function computes the softmax probabilites given
%                   given a weight matrix/vector weights and a set of
%                   training data trainingData    
    
    probabilityVector = exp( weights.' * trainingData.' ) / sum( exp( ...
        weights.' * trainingData.' ) );        

end

