function [Xtrain, Ytrain] = getWindowedDataset(df, window, trainable)

length = height(df) - window + 1;

% compose Xtrain, Ytrain matrix
% Ytrain matrix should be one-hot encoded

close = df.close;

close(1) = 0;
for i = 2:numel(close)
    close(i) = df.close(i) - df.close(i-1);
end

close = (close - mean(close)) / std(close);

Xtrain = zeros(window, length);
Ytrain = [];

for i = 1:length
    Xtrain(:, i) = double(close(i:i+window-1));
end

if trainable
    Ytrain = zeros(3, length);

    for i = 1:length
        labels = double(df.label(i:i+window-1));
        Ytrain(:, i) = onehot(labels(end)+1, 3);
    end
end

end

