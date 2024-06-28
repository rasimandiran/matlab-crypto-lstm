function y = onehot(x, numClasses)

y = zeros(numClasses, 1);
y(x) = 1;

end