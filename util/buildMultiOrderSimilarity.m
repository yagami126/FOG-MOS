function S = buildMultiOrderSimilarity(W, O)
n = size(W, 1);
S = zeros(n);
P = W;

for o = 1:O
    S = S + P;
    if o < O
        P = P * W;
    end
end

S = S - diag(diag(S));
s_min = min(S(:));
s_max = max(S(:));

if s_max > s_min
    S = (S - s_min) / (s_max - s_min);
else
    S = zeros(n);
end
end
