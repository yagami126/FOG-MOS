function prediction = predictWithOneNearestNeighbor(Xs, Xt, Ys)
%PREDICTWITHONENEARESTNEIGHBOR Predict target labels using Euclidean 1-NN.

Ys = Ys(:);
if size(Xs, 2) ~= numel(Ys)
    error('FOGMOS:LabelCountMismatch', ...
        'The number of source labels must equal the source sample count.');
end

distance = squaredEuclideanDistance(Xs, Xt);
[~, nearestSourceIndex] = min(distance, [], 1);
prediction = Ys(nearestSourceIndex(:));
end
