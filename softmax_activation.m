function [ probability_vector ] = softmax_activation( theta, x )
%softmax_activiation This function computes the softmax probabilites given
%                    given a weight vector theta and data vector x
    
    probability_vector = exp( theta.' * x.' ) / sum( exp( theta.' * x.' ) );        

end

