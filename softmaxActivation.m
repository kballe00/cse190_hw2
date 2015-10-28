function [ probabilityVector ] = softmaxActivation( layerInput )
%softmaxActiviation This function computes the softmax probabilites given
%                   given a weight matrix/vector weights and a set of
%                   training data trainingData    
    
    probabilityVector = exp( layerInput ) / sum( exp( ...
        layerInput ) );        

end

