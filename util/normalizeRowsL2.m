function X = normalizeRowsL2(X)
X = X ./ repmat(1e-4 + sqrt(sum(X.^2, 2)), [1, size(X, 2)]);
end
