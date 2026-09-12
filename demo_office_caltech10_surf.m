clear;
clc;

root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root, 'util'));

src_list = {'caltech', 'caltech', 'caltech', 'amazon', 'amazon', 'amazon', ...
    'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
tgt_list = {'amazon', 'webcam', 'dslr', 'caltech', 'webcam', 'dslr', ...
    'caltech', 'amazon', 'dslr', 'caltech', 'amazon', 'webcam'};

options.srm_gamma = 1;
options.srm_mu = 0.1;
options.T = 10;
options.dim = 50;
options.k = 16;
options.lambda = 0.1;
options.O = 4;
options.mu = 0.9;
options.alpha = 5;
options.gamma = 0.1;

n_task = length(src_list);
result = zeros(options.T, n_task);

for i = 1:n_task
    src = src_list{i};
    tgt = tgt_list{i};
    fprintf('%d: %s_vs_%s\n', i, src, tgt);

    src_data = load(fullfile(root, 'data', 'office+caltech_surf', ...
        [src '_SURF_L10.mat']));
    src_data.fts = src_data.fts ./ repmat(sum(src_data.fts, 2), 1, size(src_data.fts, 2));
    Xs = zscore(src_data.fts, 1)';
    Ys = src_data.labels;

    tgt_data = load(fullfile(root, 'data', 'office+caltech_surf', ...
        [tgt '_SURF_L10.mat']));
    tgt_data.fts = tgt_data.fts ./ repmat(sum(tgt_data.fts, 2), 1, size(tgt_data.fts, 2));
    Xt = zscore(tgt_data.fts, 1)';
    Yt = tgt_data.labels;

    [~, acc_ite, ~] = FOG_MOS(Xs, Ys, Xt, Yt, options);
    result(:, i) = 100 * acc_ite(:);
end

fprintf('mean accuracy is: %.2f%%\n', mean(result(options.T, :)));
