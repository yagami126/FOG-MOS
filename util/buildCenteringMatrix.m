function H = buildCenteringMatrix(n)
H = eye(n) - ones(n) / n;
end
