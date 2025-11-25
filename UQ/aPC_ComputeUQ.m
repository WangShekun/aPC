function UQ = aPC_ComputeUQ(results)
% aPC_ComputeUQ
% --------------------------------------------------------
% 基于 aPC-PCE 系数（在正交归一基下）计算全局统计量：
%   - 均值  E[Y]
%   - 方差 Var[Y]
%   - 标准差 Std[Y]
%
% 要求：
%   results.coefficients : K×1 PCE 系数向量 c_k
%   results.Indices      : K×M 多指数矩阵 alpha
%
% 在正交归一基 {φ_k} 下：
%   E[Y]   ≈ c_0
%   Var[Y] ≈ Σ_{k≠0} c_k^2
% --------------------------------------------------------

    c       = results.coefficients(:);
    Indices = results.Indices;   % K×M

    % ---- 找常数基（alpha 全 0） ----
    isConst = all(Indices == 0, 2);
    if ~any(isConst)
        error('No constant basis found (row with all-zero multi-index).');
    end
    if nnz(isConst) > 1
        warning('Multiple constant bases found, using the first one.');
    end

    idx0 = find(isConst, 1, 'first');
    c0   = c(idx0);

    % ---- 均值 ----
    meanY = c0;

    % ---- 方差：排除常数项，直接 ∑ c_k^2 ----
    maskVar      = true(size(c));
    maskVar(idx0)= false;
    c_eff        = c(maskVar);
    VarY         = sum(c_eff.^2);

    % ---- 输出 ----
    UQ.mean = meanY;
    UQ.var  = VarY;
    UQ.std  = sqrt(VarY);

    % 方便后续 Sobol 使用
    UQ.idx0   = idx0;
    UQ.VarTot = VarY;
end
