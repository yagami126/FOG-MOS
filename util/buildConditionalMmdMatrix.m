function Mc = buildConditionalMmdMatrix(Xs, Xt, Ys, Ytpseudo, C)
if size(Ys, 2) > 1
    Ys = Ys';
end
if size(Ytpseudo, 2) > 1
    Ytpseudo = Ytpseudo';
end

ns = size(Xs, 2);
nt = size(Xt, 2);
Mc = zeros(ns + nt);

for c = 1:C
    ids = find(Ys == c);
    idt = find(Ytpseudo == c);
    idx = [ids; ns + idt];
    e = zeros(length(idx), 1);
    e(1:length(ids)) = 1 / length(ids);
    e(length(ids) + 1:end) = -1 / length(idt);
    e(isinf(e)) = 0;
    Mc(idx, idx) = e * e';
end
end
