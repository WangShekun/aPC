function [data, nonConstIdx, mu, sigma] = aPC_PCE_map_to_U(X, data)
% aPC_PCE_map_to_U
% 将原始样本按训练集统计量映射到中心化 + 标准化空间：
%
%   U(:,j) = (samples(:,j) - mu(j)) / sigma(j)
%
% 输入：
%   samples   - N × M 的样本矩阵
%
% 输出：
%   U           - 标准化后的样本
%   nonConstIdx - 有效列（排除方差为零的列）
%   mu, sigma   - 每个变量的均值与标准差

    % [N, M] = size(samples); %#ok<NASGU>

    % mu    = mean(samples, 1);
    % sigma = std(samples, 0, 1);
    % 
    % sigma_safe = sigma;
    % sigma_safe(sigma_safe == 0) = 1;
    % 
    % U = (samples - mu) ./ sigma_safe;
    
    [U, mu, sigma] = zscore(X);

    nonConstIdx = sigma > 0;

    U     = U(:, nonConstIdx);
    mu    = mu(nonConstIdx);
    sigma = sigma(nonConstIdx);
    
    data.U = U;
end