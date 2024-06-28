function [Wf, Rf, bf, Wi, Ri, bi, Wg, Rg, bg, Wo, Ro, bo, V, b] = trainNetwork(app)

dataFile = app.DropDownTrainingDataset.Value;
trainRatio = app.SpinnerTrainRatio.Value;
tpRatio = app.SpinnerTPRatio.Value;
slRatio = app.SpinnerSLRatio.Value;
window = app.SpinnerWindowSize.Value;
batchSize = app.SpinnerBatchSize.Value;
H = app.SpinnerHiddenNodes.Value;
dropout = app.EditFieldDropout.Value;
lr = app.EditFieldLearningRate.Value;
mr = app.EditFieldMomentumRate.Value;
epochs = app.EditFieldEpoch.Value;

data = parquetread(fullfile("datasets", dataFile));
resampled = resampleDataDaily(data);

df = labelize(resampled, tpRatio, slRatio, window);

trainDf = df(1:floor(height(df) * trainRatio), :);
testDf = df(floor(height(df) * trainRatio):end, :);

[Xtrain, Ytrain] = getWindowedDataset(trainDf, window, true);
[Xtest, Ytest] = getWindowedDataset(testDf, window, true);

Xtrain = Xtrain ./ Xtrain(1, :);
Xtrain = (Xtrain - Xtrain(1, :));

Xtest = Xtest ./ Xtest(1, :);
Xtest = (Xtest - Xtest(1, :));

trainSize = size(Xtrain, 2);
numClasses = size(Ytrain, 1);           % number of classes

% =========================================================================
% weights and biases of LSTM
Wf = 0.01*randn(H, window);   Rf = 0.01*randn(H, H);   bf = 0.01*randn(H, 1);
Wi = 0.01*randn(H, window);   Ri = 0.01*randn(H, H);   bi = 0.01*randn(H, 1);
Wg = 0.01*randn(H, window);   Rg = 0.01*randn(H, H);   bg = 0.01*randn(H, 1);
Wo = 0.01*randn(H, window);   Ro = 0.01*randn(H, H);   bo = 0.01*randn(H, 1);
V = 0.01*randn(numClasses, H);    b = 0.01*randn(numClasses,1);

mWf = zeros(H, window);       mRf = zeros(H, H);       mbf = zeros(H, 1);
mWi = zeros(H, window);       mRi = zeros(H, H);       mbi = zeros(H, 1);
mWg = zeros(H, window);       mRg = zeros(H, H);       mbg = zeros(H, 1);
mWo = zeros(H, window);       mRo = zeros(H, H);       mbo = zeros(H, 1);
mV = zeros(numClasses, H);     mb = zeros(numClasses, 1);
% =========================================================================

batchList = 1:batchSize:trainSize;

h0 = zeros(H, 1);
c0 = zeros(H, 1);
iter = 0;

cla(app.AxesLoss);
xlim(app.AxesLoss, [1, floor(epochs*1.05)]);
xticks(app.AxesLoss, 'auto');

marker = scatter(app.AxesLoss, nan, nan, 'o', 'MarkerFaceColor', [0, 0.4470, 0.7410], 'MarkerFaceColor', [0, 0.4470, 0.7410]);
line = animatedline(app.AxesLoss, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);
grid(app.AxesLoss, 'on');

yMax = 0;
yMin = 0;

for epoch = 1:epochs
    % initialize state variables
    epochLoss = 0;

    if ~app.isTraining
        % Cleanup and return
        disp('Training stopped by user');
        pred = lstmForward(Wf, Rf, bf, Wi, Ri, bi, Wg, Rg, bg, Wo, Ro, bo, V, b, Xtest, h0, c0);

        [~, predIdx] = max(pred);
        [~, orgIdx] = max(Ytest);

        acc = (numel(predIdx) - sum(abs(predIdx-orgIdx))) / numel(predIdx);
        app.LabelTestAccuracy.Text = strcat(num2str(acc*100, '%.2f'), " %");

        return;
    end

    for idx = batchList

        endIdx = min(idx + batchSize -1, trainSize);

        input = Xtrain(:, idx:endIdx);
        target = Ytrain(:, idx:endIdx);

        [dWf, dRf, dbf, dWi, dRi, dbi, dWg, dRg, dbg, dWo, dRo, dbo, dV, db, h0, c0, loss] = ...
            lstm(Wf, Rf, bf, Wi, Ri, bi, Wg, Rg, bg, Wo, Ro, bo, V, b, input, target, h0, c0, dropout);

        mWf = mr*mWf - lr*dWf;
        mWi = mr*mWi - lr*dWi;
        mWg = mr*mWg - lr*dWg;
        mWo = mr*mWo - lr*dWo;

        mRf = mr*mRf - lr*dRf;
        mRi = mr*mRi - lr*dRi;
        mRg = mr*mRg - lr*dRg;
        mRo = mr*mRo - lr*dRo;

        mbf = mr*mbf - lr*dbf;
        mbi = mr*mbi - lr*dbi;
        mbg = mr*mbg - lr*dbg;
        mbo = mr*mbo - lr*dbo;

        mV = mr*mV - lr*dV;
        mb = mr*mb - lr*db;

        Wf = Wf + mWf;  Rf = Rf + mRf; bf = bf + mbf;
        Wi = Wi + mWi;  Ri = Ri + mRi; bi = bi + mbi;
        Wg = Wg + mWg;  Rg = Rg + mRg; bg = bg + mbg;
        Wo = Wo + mWo;  Ro = Ro + mRo; bo = bo + mbo;
        V = V + mV;     b = b + mb;

        iter = iter + 1;

        epochLoss = epochLoss + loss;

    end

    if epoch == 1
        yMax = floor(epochLoss * 1.02);
        yMin = floor(epochLoss * 0.98);
        ylim(app.AxesLoss, [yMin, yMax]);
    end

    if epochLoss < yMin
        yMin = epochLoss * 0.98;
        ylim(app.AxesLoss, [yMin, yMax]);
    end

    if epochLoss > yMax
        yMax = epochLoss * 1.02;
        ylim(app.AxesLoss, [yMin, yMax]);
    end

    set(marker, 'XData', epoch, 'YData', epochLoss);
    addpoints(line, epoch, epochLoss);
    drawnow;

    % str = strcat('epoch: ', num2str(epoch), ' loss: ', num2str(epochLoss));
    % disp(str);
end

h0 = zeros(H, 1);
c0 = zeros(H, 1);
pred = lstmForward(Wf, Rf, bf, Wi, Ri, bi, Wg, Rg, bg, Wo, Ro, bo, V, b, Xtest, h0, c0);

[~, predIdx] = max(pred);
[~, orgIdx] = max(Ytest);

acc = (numel(predIdx) - sum(abs(predIdx-orgIdx))) / numel(predIdx);

app.LabelTestAccuracy.Text = strcat(num2str(acc*100, '%.2f'), " %");

app.BtnTrain.Text = "Start";
app.BtnTrain.Icon = "assets/start.png";

app.isTraining = false;

end

