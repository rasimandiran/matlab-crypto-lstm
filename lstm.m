function [dWf, dRf, dbf, dWi, dRi, dbi, dWg, dRg, dbg, dWo, dRo, dbo, dV, db, hend, cend, loss] = ...
    lstm(Wf, Rf, bf, Wi, Ri, bi, Wg, Rg, bg, Wo, Ro, bo, V, b, inputs, targets, h0, c0, dropout)

T = size(inputs, 2);        % length of LSTM network
if (T ~= size(targets, 2))
    error('Number of input samples and desired output samples does not match.');
end

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

mask = [];

for j = 1:T
    xt = inputs(:, j);
    dt = targets(:, j);
    
    ft = logsig(Wf*xt + Rf*ht + bf);
    it = logsig(Wi*xt + Ri*ht + bi);
    gt = tanh(Wg*xt + Rg*ht + bg);
    ot = logsig(Wo*xt + Ro*ht + bo);
    
    ct = ft.*ct + it.*gt;
    ht = ot.*tanh(ct);

    if j == 1
        mask = dropoutMask(ht, dropout);
    end
    
    ht = ht.*mask ./ (1 - dropout);
    
    yt = softmax(V*ht + b);
    loss = loss + xentropy(dt, yt);
    
    c(:, j) = ct;
    h(:, j) = ht;
    
    f(:, j) = ft;
    i(:, j) = it;
    g(:, j) = gt;
    o(:, j) = ot;
    
    y(:, j) = yt;
end

loss = loss / T;

dWf = zeros(size(Wf));  dRf = zeros(size(Rf));  dbf = zeros(size(bf));
dWi = zeros(size(Wi));  dRi = zeros(size(Ri));  dbi = zeros(size(bi));
dWg = zeros(size(Wg));  dRg = zeros(size(Rg));  dbg = zeros(size(bg));
dWo = zeros(size(Wo));  dRo = zeros(size(Ro));  dbo = zeros(size(bo));
dV = zeros(size(V));    db = zeros(size(b));

dhnext = zeros(size(h0));
dcnext = zeros(size(c0));

for j = T:-1:1
    xt = inputs(:,j);
    dt = targets(:,j);
    yt = y(:,j);
    
    it = i(:,j);
    ot = o(:,j);
    ft = f(:,j);
    gt = g(:,j);
    
    ct = c(:,j);
    
    dyt = -(dt - yt);
    
    dht = V'*dyt.*mask + dhnext;
    dct = dht.*ot.*(1-ct.^2) + dcnext;
    
    dot = dht.*tanh(ct);
    dit = dct.*gt;
    dgt = dct.*it;
    
    if j==1
        dft = dct.*c0;
    else
        dft = dct.*c(:,j-1);
    end
    
    dV = dV + dyt*ht';
    db = db + dyt;
    
    dff = dft.*ft.*(1-ft);
    dii = dit.*it.*(1-it);
    dgg = dgt.*(1-gt.^2);
    doo = dot.*ot.*(1-ot);
    
    dWf = dWf + dff*xt';
    dWi = dWi + dii*xt';
    dWg = dWg + dgg*xt';
    dWo = dWo + doo*xt';
    
    dbf = dbf + dff;
    dbi = dbi + dii;
    dbg = dbg + dgg;
    dbo = dbo + doo;
    
    if j==1
        dRf = dRf + dff*h0';
        dRi = dRi + dii*h0';
        dRg = dRg + dgg*h0';
        dRo = dRo + doo*h0';
    else
        dRf = dRf + dff*h(:,j-1)';
        dRi = dRi + dii*h(:,j-1)';
        dRg = dRg + dgg*h(:,j-1)';
        dRo = dRo + doo*h(:,j-1)';
    end
    
    dcnext = dct.*ft;
    dhnext = Rf'*dft.*ft.*(1-ft) + Ri'*dit.*it.*(1-it) + ...
        Rg'*dgt.*(1-gt.^2) + Ro'*dot.*ot.*(1-ot);
    
end

hend = h(:, end);
cend = c(:, end);

dWf = dWf / T;  dRf = dRf / T;  dbf = dbf / T;
dWi = dWi / T;  dRi = dRi / T;  dbi = dbi / T;
dWg = dWg / T;  dRg = dRg / T;  dbg = dbg / T;
dWo = dWo / T;  dRo = dRo / T;  dbo = dbo / T;
dV = dV / T;    db = db / T;

end
