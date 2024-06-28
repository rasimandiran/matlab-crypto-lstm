function mask = dropoutMask(input, p)
%DROPOUTLAYER Summary of this function goes here
%   Detailed explanation goes here
mask = rand(size(input)) > p;
end

