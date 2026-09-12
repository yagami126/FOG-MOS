function acc = classificationAccuracy(Yp, Yt)
if size(Yp, 2) > 1
    Yp = Yp';
end
if size(Yt, 2) > 1
    Yt = Yt';
end
acc = sum(Yp == Yt) / length(Yt);
end
