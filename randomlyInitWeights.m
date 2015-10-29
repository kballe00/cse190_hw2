function weights = randomlyInitWeights( numOutConnections, numInConnections )
%RANDOMLYINITWEIGHTS Randomly initialize the weights of a layer with
%numInConnections incoming connections and numOutCounnections outgoing connections
%   weights = RANDOMLYINITWEIGHTS(numInConnections, numOutConnections) randomly
%   initializes the weights of a layer with numInConncections incoming connections
%   and numOutConnections outgoing connections. 
%
%   Note that weights should be set to a matrix of size( numInConnections,
%   numOutConnections + 1 ) as the column row of weights handles the "bias" terms
%

% Randomly initialize the weights to small values
epsilon = 0.10;
weights = rand( numOutConnections, numInConnections + 1 ) * 2 * ...
    epsilon - epsilon;

end

