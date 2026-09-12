function M0 = buildMarginalMmdMatrix(Xs, Xt, C)
ns = size(Xs, 2);
nt = size(Xt, 2);
e = [ones(ns, 1) / ns; -ones(nt, 1) / nt];
M0 = e * e' * C;
end
