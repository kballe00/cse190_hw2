% Read in training data
function [ mnist_images, mnist_labels ] = read_mnist_training_files( image_file, ...
    label_file, read_digits, offset )
    
    % Read in MNIST image data
    file_id = fopen( image_file, 'r', 'b' );
    file_header = fread( file_id, 1, 'int32' );
    
    if ( file_header ~= 2051 )
        error( 'Aborting: Invalid MNIST image file header. ' );
    end
    
    read_count = fread( file_id, 1, 'int32' );
    if ( read_count < ( read_digits + offset ) )
        error( 'Aborting: Attempting to read too many digits. ' );
    end
    
    num_rows = fread( file_id, 1, 'int32' );
    num_cols = fread( file_id, 1, 'int32' );
    
    if ( offset > 0 )
        fseek( file_id, num_rows * num_cols * offset, 'cof' );
    end
    
    mnist_images = zeros( [ num_rows num_cols read_digits ] );
    
    for i = 1 : read_digits
        for y = 1 : num_rows
            mnist_images( y, :, i ) = fread( file_id, num_cols, 'uint8' );
        end
    end
    
    fclose( file_id );
    
    % Read in MNIST image labels
    file_id = fopen( label_file, 'r', 'b' );
    file_header = fread( file_id, 1, 'int32' );
    
    if ( file_header ~= 2049 )
        error( 'Aborting: Invalid MNIST lable file header.' );
    end
    
    read_count = fread( file_id, 1, 'int32' );
    if ( read_count < ( read_digits + offset ) )
        error( 'Aborting: Attempting to read too many digits.' );
    end
    
    if ( offset > 0 )
        fseek( file_id, offset, 'cof' );
    end
    
    mnist_labels = fread( file_id, read_digits, 'uint8' );
    
    fclose( file_id );
    
    % Normalize pizel values
    mnist_images = normalize_digits( mnist_images );
    
end

function normalized_digits = normalize_digits( digits )
    normalized_digits = double( digits );
    
    for i = 1 : size( digits, 3 )
        normalized_digits( :, :, i ) = digits( :, :, i ) ./ 255.0;
    end
end
    