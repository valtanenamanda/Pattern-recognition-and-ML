%% CNN + FULLY CONNECTED -PARAMETRIT


clc; close all; clearvars;
folder = 'digits_3d\training_data';
res = 64;
files = dir(fullfile(folder, 'stroke_*.mat'));

% Remove outliers from the data
files = files(~contains({files.name}, 'REMOVED'));

n_files = length(files);
images = zeros(res, res, 2*n_files);
labels = zeros(2*n_files,1);

% Data processing
% mode 0: don't add gaussian blur
% mode 1: add gaussian blur
for mode = 0:1
    for ii = 1:n_files

        filename = files(ii).name;
        path = fullfile(folder, filename);

        load(path); % = pos

        % feature extraction (remove the 3rd dimension)
        pos2d = pos(:,1:2);

        % make datapoints into images
        img = points2images(pos2d, res);

        % add gaussian blur to second set of data
        if mode == 1
            img = add_gaussian_blur(img);
        end

        images(:, :, ii+ n_files*mode) = img;

        % true class from filename
        filename_parts = strsplit(filename, {'_', '.'}); % katkaistaan _ ja . kohdalta palasiin
        digit = str2double(filename_parts{2});
        labels( ii+ n_files*mode) = digit;
    end
end


%% train test split

n_images = length(images);
n_orig = n_files;                    % number of original (non-blurred) images
orig_idx = 1:n_orig;                 % indices of original images
blur_idx = n_orig+1 : 2*n_orig;      % indices of blurred images

% Random permutation of original image indices
perm_orig = randperm(n_orig);

% 20% of ORIGINAL images for test
n_test_orig = floor(0.2 * n_orig);
test_idx_orig  = perm_orig(1:n_test_orig);          % these are ONLY non-blurred images
train_idx_orig = perm_orig(n_test_orig+1:end);      % remaining originals

% Training set: remaining originals + all blurred images
train_idx_all = [train_idx_orig, blur_idx];

% Optional: shuffle training indices
train_idx_all = train_idx_all(randperm(numel(train_idx_all)));

% Build train and test sets

% train set
train_images = images(:,:,train_idx_all);
train_labels = labels(train_idx_all);

% test set (only clean originals, no blur)
test_images  = images(:,:,test_idx_orig);
test_labels  = labels(test_idx_orig);

%% NEURAL NETWORK code credits: 
% https://se.mathworks.com/matlabcentral/fileexchange/118545-handwritten-digit-recognition-with-simple-neural-net
% Convolution part credits:
% http://neuralnetworksanddeeplearning.com/chap6.html
% https://www.geeksforgeeks.org/computer-vision/backpropagation-in-convolutional-neural-networks/

%% PARAMETERS:

k = 5;             % Convolution filter size: 5x5
nFilters = 4;      % Number of convolution filters

conv_out_h = res - k + 1;
conv_out_w = res - k + 1;

% 2x2 average pooling 
pool_h = floor(conv_out_h/2);  
pool_w = floor(conv_out_w/2);

% Flattened feature vector dimension:
% all pooled feature maps concatenated
conv_out_dim = pool_h * pool_w * nFilters;

% Fully connected layer sizes
hidden_layer_units  = 80;   % Number of neurons in the hidden layer
output_layer_units  = 10;   % Number of classes (digits 0–9)

% 5x5 convolution filters
K = randn(k, k, nFilters) * 0.1;
% Convolutional biases, one bias per filter
b_conv = zeros(nFilters, 1);


% Weights from conv feature vector to hidden layer
W1 = randn(hidden_layer_units, conv_out_dim) * 0.1;
b1 = randn(hidden_layer_units, 1) * 0.1;

% Weights from hidden layer to output layer
W2 = randn(output_layer_units, hidden_layer_units) * 0.1;
b2 = randn(output_layer_units, 1) * 0.1;

%% TRAINING LOOP

N_epochs = 10;            % Number of training epochs
learning_rate = 0.05;     
n_samples = numel(train_labels);

% Store total loss per epoch
loss_list = zeros(1, N_epochs);

% Store the number of misclassified samples per epoch errors_list = zeros(1, N_epochs);
errors_list = zeros(1, N_epochs);

for epoch = 1:N_epochs
    % Random order of training samples for this epoch
    example_list = randperm(n_samples);
    epoch_loss = 0;   % accumulate loss over the epoch
    errors = 0; % Count training errors in this epoch

    for k_i = 1:n_samples
        % Current training sample index (shuffled)
        idx = example_list(k_i);

        % Forward pass:

        % Extract single 64x64 training image
        img = train_images(:, :, idx);

        % Convolution + ReLU + 2x2 average pooling + flatten
        [feat_vec, conv_maps] = conv_forward(img, K, b_conv);

        % Fully connected hidden layer with sigmoid activation
        Y = sigmoid(W1 * feat_vec + b1);  

        % Output layer with softmax 
        O = softmax(W2 * Y + b2);         

        % Prediction and training error counting:

        % Find index of maximum output (1–10)
        [~, answer] = max(O);               % 1–10
        answer = answer - 1;                % 0–9

        % True label (0–9)
        true_label = train_labels(idx);  

        % Track error if predicted label does not match the true one
        if answer ~= true_label
            errors = errors + 1;
        end

        % Training loss (cross-entropy)
        eps_val = 1e-12;          % prevent log(0)
        epoch_loss = epoch_loss - log(O(true_label+1) + eps_val);

        % Backpropagation:

        % Construct one-hot target vector of length 10
        t_i = zeros(output_layer_units,1);
        t_i(true_label+1) = 1;

        % Output layer error with cross-entropy loss + softmax
        delta2 = O - t_i; 

        % Hidden layer error
        dY = Y .* (1 - Y);
        delta1 = (W2' * delta2) .* dY;

        % Gradients for W2 and b2
        deltaW2 = delta2 * Y';
        deltab2 = delta2;

        % Gradients for W1 and b1
        deltaW1 = delta1 * feat_vec';
        deltab1 = delta1;

        % Gradients for convolution layer: 

        dFeat = W1' * delta1;   % gradient of the loss with respect to the feature vector
        % Backprop through pooling, ReLU, and convolution
        [dK, db_conv] = conv_backward(img, conv_maps, dFeat, K);

        % Parameter update (fully connected & convolution layer):
        W2 = W2 - learning_rate * deltaW2;
        b2 = b2 - learning_rate * deltab2;

        W1 = W1 - learning_rate * deltaW1;
        b1 = b1 - learning_rate * deltab1;

        K = K - learning_rate * dK;
        b_conv = b_conv - learning_rate * db_conv;
    end

    % Store number of errors for this epoch
    errors_list(epoch) = errors;

    loss_list(epoch) = epoch_loss / n_samples;   % average loss

    % Compute training accuracy
    acc = 1 - errors/n_samples;
    fprintf('Epoch %d/%d, errors: %d / %d, accuracy: %.2f %%\n', ...
        epoch, N_epochs, errors, n_samples, acc*100);
end

figure;
plot(1:N_epochs, loss_list, '-o', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Training loss (cross-entropy)');
title('Training Loss per Epoch');
grid on;

%% Testing

n_test = numel(test_labels);
test_errors = 0;
confmat = zeros(10, 10);

for i = 1:n_test
    img = test_images(:, :, i);

    % Forward pass only
    [feat_vec, ~] = conv_forward(img, K, b_conv);
    Y = sigmoid(W1 * feat_vec + b1);
    O = softmax(W2 * Y + b2);

    % Predicted class (0–9)
    [~, pred] = max(O);
    pred = pred - 1;      % 0–9

    true_label = test_labels(i);

    % Count misclassification
    if pred ~= true_label
        test_errors = test_errors + 1;
    end
    
    % Update confusion matrix
    confmat(true_label+1, pred+1) = confmat(true_label+1, pred+1) + 1;
end

% Test accuracy
test_acc = 1 - test_errors/n_test;

fprintf('\nTest errors: %d / %d\n', test_errors, n_test);
fprintf('Test accuracy: %.2f %%\n', test_acc*100);

% Plot confusion matrix
figure;
imagesc(confmat);
xlabel('Predicted class');
ylabel('True class');
title('Confusion matrix');
colorbar;
axis equal tight;

xticks(1:10); xticklabels(0:9);
yticks(1:10); yticklabels(0:9);

for r = 1:10
    for c = 1:10
        value = confmat(r,c);
        if value > 0
            text(c, r, sprintf('%d', value), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', ...
                'Color', 'white', ...
                'FontWeight', 'bold');
        end
    end
end

%% Helper functions

% Sigmoid activation function
function Y = sigmoid(X)
    Y = 1./(1+exp(-X));
end

% Softmax activation for output layer (column vector version)
function y = softmax(x)
    % subtract max for numerical stability
    x = x - max(x);
    ex = exp(x);
    y  = ex / sum(ex);
end

% Forward pass through convolution + ReLU + 2x2 average pooling
function [feat_vec, conv_maps] = conv_forward(img, K, b_conv)
    % Input:
    %   img      : 2D input image (H x W)
    %   K        : convolution filters (kH x kW x nFilters)
    %   b_conv   : bias vector for filters (nFilters x 1)
    % Outputs:
    %   feat_vec : flattened pooled feature maps (vector)
    %   conv_maps: raw convolution outputs (before ReLU)

    [H, W] = size(img);
    [kH, kW, nFilters] = size(K);

    % Dimensionality after valid convolution
    conv_out_h = H - kH + 1;
    conv_out_w = W - kW + 1;

    % Dimensionality after 2x2 average pooling
    pool_h = floor(conv_out_h/2);
    pool_w = floor(conv_out_w/2);

    % Allocate arrays for convolution and activation maps
    conv_maps = zeros(conv_out_h, conv_out_w, nFilters);
    pooled    = zeros(pool_h, pool_w, nFilters);

    for f = 1:nFilters
        % Convolution (valid) + bias
        % rot90(K,2) rotates the filter by 180 degrees (kernel flip)
        conv_map = conv2(img, rot90(K(:,:,f), 2), 'valid') + b_conv(f);
        conv_maps(:,:,f) = conv_map;

        % ReLU
        act = max(conv_map, 0);

        % 2x2 average pooling
        P = zeros(pool_h, pool_w);
        for i = 1:pool_h
            for j = 1:pool_w
                % Take 2x2 block from the activation map
                block = act(2*i-1:2*i, 2*j-1:2*j);
                % Store the mean of the block
                P(i,j) = mean(block, 'all');
            end
        end
        pooled(:,:,f) = P;
    end

    % Flatten all pooled feature maps into a single column vector
    feat_vec = reshape(pooled, [], 1);
end

% Backpropagation for convolutional layer (average pooling + ReLU + conv)
function [dK, db_conv] = conv_backward(img, conv_maps, dFeat, K)
    % Input:
    %   img      : original input image (H x W)
    %   conv_maps: convolution outputs (before ReLU) for each filter
    %   dFeat    : gradient of the loss with respect to the feature vector
    %   K        : convolution filters (for dimensions)
    %
    % Outputs:
    %   dK       : gradient w.r.t. filters K
    %   db_conv  : gradient w.r.t. conv biases

    [H, W] = size(img);
    [kH, kW, nFilters] = size(K);

    conv_out_h = H - kH + 1;
    conv_out_w = W - kW + 1;
    
    % Pooling size
    pool_h = floor(conv_out_h/2);
    pool_w = floor(conv_out_w/2);

    % Initialize gradients for filters and biases
    dK = zeros(size(K));
    db_conv = zeros(nFilters, 1);
    
    % dFeat contains gradients for all pooled feature maps (flattened).
    % Reshape once into [pool_h, pool_w, nFilters]
    dP_all = reshape(dFeat, [pool_h, pool_w, nFilters]);
  
    for f = 1:nFilters
        dP = dP_all(:,:,f);

        % Backprop 2x2 average pooling
        % Each pooled value is the mean of a 2x2 block,
        % so each element in that block gets 1/4 of the gradient.
        dAct = zeros(conv_out_h, conv_out_w);
        for i = 1:pool_h
            for j = 1:pool_w
                grad = dP(i,j) / 4;
                dAct(2*i-1:2*i, 2*j-1:2*j) = dAct(2*i-1:2*i, 2*j-1:2*j) + grad;
            end
        end

        % Backprop ReLU
        % ReLU derivative: 1 if conv_map > 0, else 0
        conv_map = conv_maps(:,:,f);
        mask = conv_map > 0;
        dConv = dAct .* mask;

        % Gradient for bias 
        db_conv(f) = sum(dConv, 'all');

        % --- Gradient for filter K(:,:,f) ---
        % Forward: conv2(img, rot90(K), 'valid')
        % Backward: conv2(img, rot90(dConv), 'valid')
        dK(:,:,f) = conv2(img, rot90(dConv, 2), 'valid');
    end
end
