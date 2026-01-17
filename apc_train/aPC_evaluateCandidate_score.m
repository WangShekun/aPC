function out = aPC_evaluateCandidate_score(data, ModelOpts, map_U)
% 并行候选评估：返回 out.score（用于搜索）
% SearchMetric:
%   - 'loo'    : score = LOO
%   - 'relmse' : score = RelMSE on validation set (calls aPC_Predict)

% —— 拷贝一份，避免修改外部结构 —— 
data_loc      = data;
Modelopts_loc = ModelOpts;

% ---------- 搜索准则 ----------
if ~isfield(Modelopts_loc, 'SearchMetric') || isempty(Modelopts_loc.SearchMetric)
    SearchMetric = 'loo';
else
    SearchMetric = lower(Modelopts_loc.SearchMetric);
end

% —— 降维：如未指定 red_dim 则跳过 —— 
if isfield(Modelopts_loc,'red_dim') && ~isempty(Modelopts_loc.red_dim)
    [data_loc, Modelopts_loc, ReduceModel] = aPC_dimReduction(data_loc, Modelopts_loc);
else
    ReduceModel = [];
    if ~isfield(Modelopts_loc,'num_vars') || isempty(Modelopts_loc.num_vars)
        Modelopts_loc.num_vars = size(data_loc.U,2);
    end
end

% —— 多索引 & 基函数 —— 
Modelopts_loc.Indices = aPC_GenerateTerms(Modelopts_loc);
[Psi_tmp, poly_tmp, ~] = aPC_PCE_build_Psi(Modelopts_loc, data_loc);

% —— 回归 —— 
res_tmp = aPC_RegressionMethods(Psi_tmp, data_loc.Y_train, Modelopts_loc.RegressionMethod);

% —— 评分 —— 
switch SearchMetric
    case 'loo'
        score = res_tmp.LOO;

    case 'relmse'
        if ~isfield(data,'X_val') || ~isfield(data,'Y_val')
            error('[aPC_evaluateCandidate_LOO] relmse requires data.X_val and data.Y_val');
        end

        % ===== 补齐预测所需字段（不污染 data）=====
        res_tmp.map_U       = map_U;
        res_tmp.poly_coeffs = poly_tmp;
        res_tmp.Indices     = Modelopts_loc.Indices;

        Yv_pred = aPC_Predict(data.X_val, res_tmp, Modelopts_loc, ReduceModel);
        yv = data.Y_val;

        score = sum((Yv_pred - yv).^2) / max(sum((yv - mean(yv)).^2), eps);

    otherwise
        error('[aPC_evaluateCandidate_LOO] Unknown SearchMetric: %s', SearchMetric);
end

out.score = score;
end
