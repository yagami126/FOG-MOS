function S = learnFlexibleGraph(X, P, rrr)
n = size(X, 2);
D = squaredEuclideanDistance(X, X);
S = zeros(n);

for i = 1:n
    d = D(i, :);
    p = P(i, :);
    v = -(d - 2 * rrr * p) / (2 * rrr);
    S(i, :) = projectOntoSimplex(v);
end

S = (S + S') / 2;
end
