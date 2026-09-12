function S = buildMultiOrderGraph(X, k, O)
W = buildCosineKnnGraph(X', k);
S = buildMultiOrderSimilarity(W, O);
end
