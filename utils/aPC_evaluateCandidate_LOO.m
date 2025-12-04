function out = aPC_evaluateCandidate_LOO(data, ModelOpts)
% 给定一个候选模型参数组合，快速评估其 LOO 误差（用作自动搜索的轻量预评估）。

% —— 拷贝一份，避免修改外部结构 —— 
data_loc = data;
Modelopts_loc   = ModelOpts;

% —— 降维：如未指定 red_dim 则跳过 —— 
if isfield(Modelopts_loc,'red_dim') && ~isempty(Modelopts_loc.red_dim)
    [data_loc, Modelopts_loc, ~] = aPC_dimReduction(data_loc, Modelopts_loc);
else
    if ~isfield(Modelopts_loc,'num_vars') || isempty(Modelopts_loc.num_vars)
        Modelopts_loc.num_vars = size(data_loc.U,2);
    end
end

% —— 多索引 & 基函数 —— 
Modelopts_loc.Indices = aPC_GenerateTerms(Modelopts_loc);
[Psi_tmp, ~, ~] = aPC_PCE_build_Psi(Modelopts_loc, data_loc);

% —— 回归 —— 
res_tmp = aPC_RegressionMethods(Psi_tmp, data_loc.Y_train, Modelopts_loc.RegressionMethod);

% —— 输出（尽量轻量） —— 
out.LOO    = res_tmp.LOO;
% out.RelMSE = [];
% 也可把组合回传便于调试
% out.choice = Mo_loc;
end
