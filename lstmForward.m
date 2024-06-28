function y = lstmForward(Wf, Rf, bf, Wi, Ri, bi, Wg, Rg, bg, Wo, Ro, bo, V, b, inputs, h0, c0)

T = size(inputs, 2);        % length of LSTM network

H = size(Rf, 1);    % number of hidden nodes
D = size(V, 1);   % number of output nodes

h = zeros(H, T);
c = zeros(H, T);
f = zeros(H, T);
i = zeros(H, T);
g = zeros(H, T);
o = zeros(H, T);
y = zeros(D, T);

ht = h0;
ct = c0;
loss = 0;

for j = 1:T
    xt = inputs(:, j);
    
    ft = logsig(Wf*xt + Rf*ht + bf);
    it = logsig(Wi*xt + Ri*ht + bi);
    gt = tanh(Wg*xt + Rg*ht + bg);
    ot = logsig(Wo*xt + Ro*ht + bo);
    
    ct = ft.*ct + it.*gt;
    ht = ot.*tanh(ct);
    
    yt = softmax(V*ht + b);
    
    c(:, j) = ct;
    h(:, j) = ht;
    
    f(:, j) = ft;
    i(:, j) = it;
    g(:, j) = gt;
    o(:, j) = ot;
    
    y(:, j) = yt;
end

end
