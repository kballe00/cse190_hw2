function numericalGradient = computeNumericalGradient( costFunction, weights )
%COMPUTENUMERICALGRADIENT Computes the gradient using finite differences
%and gives a numerical estimate of the gradient.
%   numericalGradient = COMPUTENUMERICALGRADIENT( costFunction, weights )
%   computes the numerical gradient of the function costFunction using weights.
%   Calling y = costFunction( weights ) should return the function value at theta.         

% Initialize variables
numericalGradient = zeros( size( weights ) );
epislonVector = zeros( size( weights ) );
epislon = 1e-4;

for i = 1 : numel( theta )
    % Set appropiate index in epsilonVector
    epislonVector( i ) = epislon;
    loss1 = costFunction( weights - epsilonVector );
    loss2 = costFunction( weights + epislonVector );
    
    % Compute the numerical gradient
    numericalGradient( i ) = ( loss2 - loss1 ) / ( 2 * epislon );
    epislonVector( i ) = 0;
end

end
