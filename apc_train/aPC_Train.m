function [results, ModelOpts, ReduceModel] = aPC_Train(data, ModelOpts)
% aPC_Train
% 封装 aPC 训练流程（标准化 → 降维 → 基函数 → 回归）

%% ---------- 1) 标准化 ----------
[data, nonConstIdx, mu, sigma] = aPC_PCE_map_to_U(data.X_train, data);

%% ---------- 2) 降维 ----------
[data, ModelOpts, ReduceModel] = aPC_dimReduction(data, ModelOpts);

%% ---------- 3) 多项式指数 ----------
ModelOpts.Indices = aPC_GenerateTerms(ModelOpts);

%% ---------- 4) 构造基函数 ----------
[Psi, poly_coeffs, univ_vals] = aPC_PCE_build_Psi(ModelOpts, data);

%% ---------- 5) 回归 ----------
results = aPC_RegressionMethods(Psi, data.Y_train, ModelOpts.RegressionMethod);

%% ---------- 6) 保存模型信息 ----------
results.mu          = mu;
results.sigma       = sigma;
results.poly_coeffs = poly_coeffs;
results.Indices     = ModelOpts.Indices;
results.univ_vals   = univ_vals;

%% ---------- 7) 保存数据统计 ----------
results.N_train = size(data.X_train,1);

% 有验证集就存
if isfield(data, "X_val")
    results.N_test = size(data.X_val,1);
else
    results.N_test = [];
end

end
