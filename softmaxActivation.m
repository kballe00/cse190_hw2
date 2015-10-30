function [ probabilityVector ] = softmaxActivation( layerInput )
%softmaxActiviation This function computes the softmax probabilites given
%                   given a weight matrix/vector weights and a set of
%                   training data trainingData   

    probabilityVector = bsxfun( @rdivide, exp( layerInput ), sum( exp( ...
        layerInput ), 2 ) );        

end

