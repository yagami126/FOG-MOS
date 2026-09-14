function W = buildCosineKnnGraph(X, k)
n = size(X, 1);
x_norm = sqrt(sum(X.^2, 2));

for i = 1:n
    X(i, :) = X(i, :) ./ max(1e-12, x_norm(i));
end

S = full(X * X');
[s, idx] = sort(-S, 2);
idx = idx(:, 1:k + 1);
s = -s(:, 1:k + 1);

row = repmat((1:n)', [k + 1, 1]);
W = sparse(row, idx(:), s(:), n, n);
W = max(W, W');
end
