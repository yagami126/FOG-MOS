function D = squaredEuclideanDistance(X, Y)
if size(X, 1) == 1
    X = [X; zeros(1, size(X, 2))];
    Y = [Y; zeros(1, size(Y, 2))];
end

XX = sum(X .* X);
YY = sum(Y .* Y);
XY = X' * Y;
D = repmat(XX', [1, size(YY, 2)]) ...
    + repmat(YY, [size(XX, 2), 1]) - 2 * XY;
D = max(real(D), 0);
end
