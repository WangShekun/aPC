function aPC_printUQ(UQ)
% aPC_print_UQ
% -------------------------------------------
% 打印不确定性量化（UQ）结果：
%   - mean, variance, std
%   - CI95
%   - skewness / kurtosis
%   - VaR95 / CVaR95
%
% 本函数只输出 “不确定性相关量”，
% 不涉及模型误差，也不使用 y_val。
% -------------------------------------------

line        = repmat('-', 1, 70);
line_double = repmat('=', 1, 70);

fprintf('\n%s\n', line_double);
fprintf('Uncertainty Quantification (UQ) Report\n');
fprintf('%s\n', line);

%% =====================================================
%% 1) 解析 UQ（来自 PCE 系数）
%% =====================================================
fprintf('aPC-based analytic statistics:\n');
fprintf('  Mean (E[Y])             : %12.6e\n', UQ.mean);
fprintf('  Variance (Var[Y])       : %12.6e\n', UQ.var);
fprintf('  Std deviation (Std[Y])  : %12.6e\n', UQ.std);
fprintf('%s\n', line);

%% =====================================================
%% 2) 分布形状（来自 y_{pred} ）
%% =====================================================
fprintf('Distribution shape (from predicted samples):\n');
fprintf('  Skewness                : %12.6e\n', UQ.skew);
fprintf('  Kurtosis                : %12.6e\n', UQ.kurt);
fprintf('%s\n', line);

%% =====================================================
%% 3) 置信区间（CI）
%% =====================================================
fprintf('Confidence Intervals (from predicted samples):\n');
fprintf('  95%% CI  [2.5%%, 97.5%%] : [%12.6e , %12.6e]\n', ...
         UQ.CI95(1), UQ.CI95(2));
fprintf('%s\n', line);

%% =====================================================
%% 4) 风险度量（VaR / CVaR）
%% =====================================================
fprintf('Tail risk metrics:\n');
fprintf('  VaR95 (95%% quantile)   : %12.6e\n', UQ.VaR95);
fprintf('  CVaR95 (tail mean)      : %12.6e\n', UQ.CVaR95);
fprintf('%s\n', line_double);

end
