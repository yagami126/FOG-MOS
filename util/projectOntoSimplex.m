function x = projectOntoSimplex(v)
n = length(v);
x = v - mean(v) + 1 / n;

if min(x) < 0
    f = 1;
    lambda = 0;
    iter = 1;
    while abs(f) > 10^-10
        z = x - lambda;
        idx = z > 0;
        df = -sum(idx);
        f = sum(z(idx)) - 1;
        lambda = lambda - f / df;
        iter = iter + 1;
        if iter > 100
            x = max(z, 0);
            return;
        end
    end
    x = max(z, 0);
end
end
