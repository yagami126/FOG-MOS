function Ytpseudo = predictWithSRM(Xs, Xt, Ys, options)
X = [Xs, Xt];
ns = size(Xs, 2);
nt = size(Xt, 2);
C = length(unique(Ys));

K = rbfKernel(X, options.gamma);
J = diag([ones(ns, 1); zeros(nt, 1)]);
Y = [oneHotLabels(Ys, C); zeros(nt, C)];
n = size(K, 1);
B = (J * K + options.mu * eye(n)) \ (J * Y);
score = B' * K;
[~, Ytpseudo] = max(score(:, ns + 1:end), [], 1);
Ytpseudo = Ytpseudo';
end

function K = rbfKernel(X, gamma)
XX = sum(X.^2, 1);
n = size(X, 2);
D = (ones(n, 1) * XX)' + ones(n, 1) * XX - 2 * (X' * X);
K = exp(-gamma * D);
end

function Y = oneHotLabels(label, C)
n = length(label);
Y = zeros(n, C);
for i = 1:n
    if label(i) > 0 && label(i) <= C
        Y(i, label(i)) = 1;
    end
end
end
