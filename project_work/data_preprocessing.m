%%
close all; clearvars; clc;

folder = 'digits_3d\training_data';
res = 64;
files = dir(fullfile(folder, 'stroke_*.mat'));

n_files = length(files);
images = zeros(res, res, n_files);
labels = zeros(n_files,1);

for ii = 1:n_files

    filename = files(ii).name;
    path = fullfile(folder, filename);

    load(path); % = pos

    % feature extraction (poistetaan 3 ulottuvuus)
    pos2d = pos(:,1:2);

    % tehdään datapisteistä kuvia
    img = points2images(pos2d, res);

    images(:, :, ii) = img;

    % luokka saadaan tiedostonimestä
    filename_parts = strsplit(filename, {'_', '.'}); % katkaistaan _ ja . kohdalta palasiin
    digit = str2double(filename_parts{2});
    labels(ii) = digit;

end

for dig = 0:9

    figure
    for k = 1:9
        subplot(3,3,k)
        imshow(images(:,:,k+dig*100), [])
        title(sprintf("Label = %d", labels(k+dig*100)))
    end

end

%% train test split

idx = randperm(n_files);

% 80% train + 20% test
n_train = floor(0.8 * n_files);
train_idx = idx(1:n_train);
test_idx  = idx(n_train+1:end);

% train set
train_images = images(:,:,train_idx);
train_labels = labels(train_idx);

% test set
test_images  = images(:,:,test_idx);
test_labels  = labels(test_idx);

% reshape training and testing images as vectors
X_train = reshape(train_images, 64*64, size(train_images,3));
X_test = reshape(test_images, 64*64, size(test_images,3));
%% NEURAL NETWORK code (credits: https://se.mathworks.com/matlabcentral/fileexchange/118545-handwritten-digit-recognition-with-simple-neural-net)

% one-hot encoding to labels (target vectors)

n_train = numel(train_labels); 
T = zeros(10, n_train); 
for i = 1:n_train 
    T(train_labels(i)+1, i) = 1; 
end 
n_test = numel(test_labels); 
T_test = zeros(10, n_test); 
for i = 1:n_test 
    T_test(test_labels(i)+1, i) = 1; 
end

%% neural network parameters
input_layer_units = 64*64;
hidden_layer_units  = 80;      
output_layer_units  = 10; 

W1 = rand(hidden_layer_units, input_layer_units)*2 - 1;
b1 = rand(hidden_layer_units, 1)*2 - 1;
W2 = rand(output_layer_units, hidden_layer_units)*2 - 1;
b2 = rand(output_layer_units, 1)*2 - 1;

%% training loop
N_epochs = 50;       
learning_rate = 0.1;     
n_samples = size(X_train, 2);
errors_list = zeros(1, N_epochs);

for epoch = 1:N_epochs

    example_list = randperm(n_samples);
    errors = 0;
    
    for k = 1:n_samples
        i = example_list(k);  
        % FORWARD
        x = X_train(:, i);                % yksi kuva (4096 x 1)
        
        % 1. layer (sigmoid)
        Y = sigmoid(W1*x+b1);             
        
        % 2. layer (sigmoid, output)
        O = sigmoid(W2*Y+b2);
        
        % answer and the correct class
        [~, answer] = max(O);                  % 1–10
        answer = answer - 1;                   % 0–9
        
        [~, correct_answer] = max(T(:, i));    % 1–10
        correct_answer = correct_answer - 1;   % 0–9
        
        if answer ~= correct_answer
            errors = errors + 1;
        end
        
        % BACKPROP
        t_i = T(:, i);   % one-hot target
        
        % delta2 (output layer), sigmoid
        
        delta2 = 2*(O - t_i) .* O .* (1 - O);
        
        
        % delta1 (hidden), sigmoid
        dY = Y .* (1 - Y);
        delta1 = (W2' * delta2) .* dY;
        
        % Gradients
        deltaW2 = delta2 * Y';
        deltab2 = delta2;
        deltaW1 = delta1 * x';
        deltab1 = delta1;
        
        % Update weights
        W2 = W2 - learning_rate * deltaW2;
        b2 = b2 - learning_rate * deltab2;
        W1 = W1 - learning_rate * deltaW1;
        b1 = b1 - learning_rate * deltab1;
    end
    
    % Epoch statistics
    errors_list(epoch) = errors;
    acc = 1 - errors/n_samples;
    fprintf('Epoch %d/%d, errors: %d / %d, accuracy: %.2f %%\n', ...
        epoch, N_epochs, errors, n_samples, acc*100);
end

figure;
plot(1:N_epochs, errors_list, '-o');
xlabel('Epoch');
ylabel('Number of errors');
grid on;
title('Learning history (errors per epoch)');

%%

test_errors = 0;
confmat = zeros(10,10);

for i = 1:n_test
    
    x = X_test(:,i);

    % forward
    Y  = sigmoid(W1*x + b1);     % sigmoid
    O  = sigmoid(W2*Y + b2);      % sigmoid

    [~, pred] = max(O);
    pred = pred - 1;           % 0–9

    true = test_labels(i);     % 0–9
    
    % virheet
    if pred ~= true
        test_errors = test_errors + 1;
    end

    % confusion matriisi
    confmat(true+1, pred+1) = confmat(true+1, pred+1) + 1;
end

test_acc = 1 - test_errors/n_test;

fprintf('\nTest errors: %d / %d\n', test_errors, n_test);
fprintf('Test accuracy: %.2f %%\n', test_acc*100);



figure;
imagesc(confmat);
xlabel('Predicted class');
ylabel('True class');
title('Confusion matrix');
colorbar;
axis equal tight;

xticks(1:10);
xticklabels(0:9);

yticks(1:10);
yticklabels(0:9);

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
%%
function Y = sigmoid(X)
    Y = 1./(1+exp(-X));

end
