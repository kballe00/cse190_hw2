% Read in data
[ images, labels ] = read_mnist_training_files( '~/Downloads/train-images-idx3-ubyte', ...
    '~/Downloads/train-labels-idx1-ubyte', 20000, 0 );

% Initialize variables
theta = ones( 785, 10 );
x = zeros( 20000, 784 );
for i = 1 : 20000
   x( i, : ) = reshape( images( :, :, i ).', 784, 1);
end
% Add 1's, for the bias, to the first column of the inputs
x = [ ones( 20000, 1 ) x ];
alpha = 0.01;
probabilites = zeros( 20000, 10 );
y = zeros( 10, 20000 );
correct_per_loop = [ 0 0 0 0 0 ];

% Set a matrix for representing 1{y^(i) = k} in gradient descent 
for i = 1 : 20000
    y( labels( i ) + 1, i ) = 1;
end

for j = 1 : 5
    for k = 0 : 9
        for i = 1 : 20000
            softmax_output = softmax_activation( theta, x( i, : ) );
            theta( : , k + 1 ) = theta( : , k + 1 ) + alpha * ( ...
                y( k + 1, i ) - softmax_output( k + 1 ) ) * ...
                x( i, : ).';
        end
    end
    
    % Test model accuracy per iteration through gradient descent
    for i = 1 : 20000
        probabilites( i, : ) = softmax_activation( theta, x( i, : ) ).';
    end

    [ ~, maxIndices ] = max( probabilites, [ ], 2 );

    for i = 1 : 20000
        if ( maxIndices( i ) - 1 ) == labels( i )
            correct_per_loop( j ) = correct_per_loop( j ) + 1;
        end
    end
end