function [acc, acc_ite, obj, features] = FOG_MOS(Xs, Ys, Xt, Yt, options)
T = options.T;
dim = options.dim;
k = options.k;
O = options.O;
rrr = options.rrr;
mu = options.M_mu;
alpha = options.alpha;
lambda = options.lambda;

Xs = normr(Xs')';
Xt = normr(Xt')';
X = [Xs, Xt];
X = normalizeRowsL2(X')';

[m, ns] = size(Xs);
nt = size(Xt, 2);
n = ns + nt;
C = length(unique(Ys));

acc = 0;
acc_ite = zeros(1, T);
obj = zeros(1, T);

Ytpseudo = predictWithSRM(Xs, Xt, Ys, options);

Ps = buildMultiOrderGraph(Xs, k, O);
Pt = buildMultiOrderGraph(Xt, k, O);
Ss = learnFlexibleGraph(Xs, Ps, rrr);
St = learnFlexibleGraph(Xt, Pt, rrr);

M0 = buildMarginalMmdMatrix(Xs, Xt, C);
H = buildCenteringMatrix(n);
right = X * H * X';

for i = 1:T
    S = blkdiag(Ss, St);
    S = S + S';
    L = computeNormalizedLaplacian(S);

    if isempty(Ytpseudo)
        M = M0;
    else
        Mc = buildConditionalMmdMatrix(Xs, Xt, Ys, Ytpseudo, C);
        M = (1 - mu) * M0 + mu * Mc;
    end

    G = alpha * (X * X') ...
        - alpha^2 * (X * ((2 * L + alpha * eye(size(L))) \ X'));
    L = L ./ norm(L, 'fro');
    G = G ./ norm(G, 'fro');
    M = M ./ norm(M, 'fro');

    left = X * M * X' + G + lambda * eye(m);
    [A, ~] = eigs(left, right, dim, 'sm');

    F = alpha * ((2 * L + alpha * eye(size(L))) \ (X' * A));
    F = F';

    AX = A' * X;
    AX = normalizeRowsL2(AX')';
    AXs = AX(:, 1:ns);
    AXt = AX(:, ns + 1:end);
    Ytpseudo = predictWithSRM(AXs, AXt, Ys, options);

    Ss = learnFlexibleGraph(F(:, 1:ns), Ps, rrr);
    St = learnFlexibleGraph(F(:, ns + 1:end), Pt, rrr);

    acc = classificationAccuracy(Ytpseudo, Yt);
    acc_ite(i) = acc;
    obj(i) = objective_value(A, F, Ss, St, Ps, Pt, X, M, ...
        alpha, rrr, lambda, ns);
    fprintf('[%2d] acc:%.4f\n', i, acc * 100);
end

if nargout > 3
    features.source = AXs;
    features.target = AXt;
    features.projection = A;
    features.targetPseudoLabels = Ytpseudo;
end
end

function L = computeNormalizedLaplacian(S)
n = size(S, 1);
D = diag(sparse(sqrt(1 ./ (sum(S) + eps))));
L = eye(n) - D * S * D;
end

function value = objective_value(W, F, Ss, St, Ps, Pt, X, M, ...
        alpha, rrr, lambda, ns)
Fs = F(:, 1:ns);
Ft = F(:, ns + 1:end);
Ds = squaredEuclideanDistance(Fs, Fs);
Dt = squaredEuclideanDistance(Ft, Ft);

distribution = trace(W' * X * M * X' * W);
graph = sum(sum(Ds .* Ss)) + sum(sum(Dt .* St));
reconstruction = alpha * norm(X' * W - F', 'fro')^2;
similarity = rrr * (norm(Ss - Ps, 'fro')^2 + ...
    norm(St - Pt, 'fro')^2);
regularization = lambda * norm(W, 'fro')^2;

value = real(distribution + graph + reconstruction + similarity + ...
    regularization);
end
