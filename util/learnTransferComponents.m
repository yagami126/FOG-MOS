function [Z, A, objective] = learnTransferComponents( ...
        X, M, lambda, dim, kernelType, gamma)
%LEARNTRANSFERCOMPONENTS Solve the TCA/JDA generalized eigenproblem.

n = size(X, 2);
H = eye(n) - ones(n) / n;
kernelType = lower(char(kernelType));

if strcmp(kernelType, 'primal')
    basis = X;
else
    basis = transferKernel(kernelType, X, gamma);
end

matrixSize = size(basis, 1);
dim = min(dim, matrixSize - 1);
if dim < 1
    error('FOGMOS:InvalidSubspaceDimension', ...
        'The TCA/JDA subspace dimension must be at least one.');
end

left = basis * M * basis' + lambda * eye(matrixSize);
right = basis * H * basis';
left = (left + left') / 2;
right = (right + right') / 2;
[A, ~] = eigs(left, right, dim, 'sm');
A = real(A);

Z = A' * basis;
Z = Z ./ max(sqrt(sum(Z.^2, 1)), eps);
objective = real(trace(A' * left * A));
end

function K = transferKernel(kernelType, X, gamma)
switch kernelType
    case 'linear'
        K = X' * X;
    case 'rbf'
        squaredNorm = sum(X.^2, 1);
        distance = squaredNorm' + squaredNorm - 2 * (X' * X);
        K = exp(-gamma * max(real(distance), 0));
    case 'sam'
        cosineSimilarity = max(min(real(X' * X), 1), -1);
        K = exp(-gamma * acos(cosineSimilarity).^2);
    otherwise
        error('FOGMOS:UnsupportedKernel', ...
            ['Unsupported kernel "%s". Use "primal", "linear", ' ...
             '"rbf", or "sam".'], kernelType);
end
end
