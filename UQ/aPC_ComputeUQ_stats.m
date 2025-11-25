function UQ = aPC_ComputeUQ_stats(UQ, Data)
% aPC_ComputeUQ_stats
% --------------------------------------------------------
% 在已有解析 UQ 的基础上（mean,var,std），
% 利用预测样本 Y_pred 计算：
%   - 95% 置信区间 CI95
%   - 偏度 skewness
%   - 峰度 kurtosis
%   - VaR95, CVaR95（右尾风险）
%
% 输入：
%   UQ   : 已含 mean, var, std 的结构体（来自 aPC_ComputeUQ）
%   Data : 结构体，至少包含：
%          Data.Y_pred : N×1 样本（aPC 预测）
% --------------------------------------------------------

    y = Data.Y_pred(:);
    if numel(y) < 10
        warning('Too few samples in Data.Y_pred (<10), statistics may be unreliable.');
    end

    % 95% 置信区间
    UQ.CI95 = quantile(y, [0.025 0.975]);

    % 偏度 & 峰度
    UQ.skew = skewness(y);
    UQ.kurt = kurtosis(y);

    % VaR95（右尾 95% 分位）和 CVaR95（该尾部的均值）
    alpha    = 0.95;
    UQ.VaR95 = quantile(y, alpha);

    tail = y(y >= UQ.VaR95);
    if isempty(tail)
        UQ.CVaR95 = UQ.VaR95;
    else
        UQ.CVaR95 = mean(tail);
    end
end
