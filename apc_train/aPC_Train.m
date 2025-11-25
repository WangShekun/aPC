function [results, ModelOpts, data, ReduceModel] = aPC_Train(data, ModelOpts)
% aPC_Train
% ------------------------------------------------------------
% 标准化 →（可选：内部降维）→ 生成多索引(取决于 Order,qnorm)
% → 构造Ψ → 回归（由 RegressionMethod 决定）→ 以 LOO 选最优组合
%
% 兼容：
%   - ModelOpts.red_dim  可以是标量或向量（或缺省）
%   - ModelOpts.Order    可以是标量或向量
%   - ModelOpts.qnorm    可以是标量或向量
%
% 选择准则：以 aPC_RegressionMethods 返回的 results.LOO 最小者为最优
% 降维内部采用 aPC_dimReduction（不关心具体方法：PLS/PCA/… 皆在其内）
% ------------------------------------------------------------

%% ---------- 0) 默认设置 ----------
if ~isfield(ModelOpts, 'RegressionMethod') || isempty(ModelOpts.RegressionMethod)
    ModelOpts.RegressionMethod = 'LARS';  % 'LARS' | 'OLS' | 'Ridge' | 'Lasso'
end

if ~isfield(ModelOpts, 'qnorm') || isempty(ModelOpts.qnorm)
    ModelOpts.qnorm = 1;
end

%% ---------- 1) 标准化（外部物理空间 → U） ----------
[data, ~, mu, sigma] = aPC_PCE_map_to_U(data.X_train, data);
% 之后所有候选组合均在同一份“标准化后的 data（含U）”上试探

%% ---------- 2) 构造候选集合（标量→单点；向量→网格搜索） ----------
rd_list   = to_list(getfield_or_default(ModelOpts, 'red_dim', []));  %#ok<GFLD>
order_list= to_list(ModelOpts.Order);
q_list    = to_list(ModelOpts.qnorm);

% 若不设 red_dim（空），视为“不做降维”的唯一候选
if isempty(rd_list), rd_list = NaN; end

%% ---------- 3) 三重（或更少）循环：rd × p × q 以 LOO 选最优 ----------
best = struct('LOO', Inf);
ReduceModel = [];

for rd = rd_list
    for p = order_list
        for qn = q_list

            % --- 临时选项 ---
            Mo = ModelOpts;
            Mo.Order = p;
            Mo.qnorm = qn;

            % --- 降维：若 rd 为数值（且非NaN）则启用；否则跳过 ---
            if ~isnan(rd)
                Mo.red_dim = rd;
                [data_tmp, Mo, Red_tmp] = aPC_dimReduction(data, Mo);
            else
                data_tmp = data;
                Red_tmp  = [];
                % 对于“无降维”的场景，确保 num_vars 至少与当前输入维度一致
                if ~isfield(Mo,'num_vars') || isempty(Mo.num_vars)
                    if isfield(data_tmp,'U'), Mo.num_vars = size(data_tmp.U,2); end
                end
            end

            % --- 生成多索引（依赖 p 与 qn，并隐式依赖 Mo.num_vars）---
            Mo.Indices = aPC_GenerateTerms(Mo);

            % --- 构造 Ψ（在 data_tmp 上，内部会取 U 或降维后的变量）---
            [Psi_tmp, poly_tmp, univ_tmp] = aPC_PCE_build_Psi(Mo, data_tmp);

            % --- 回归，取 LOO 作为选择准则 ---
            res_tmp = aPC_RegressionMethods(Psi_tmp, data_tmp.Y_train, Mo.RegressionMethod);

            % --- 更新最优 ---
            if res_tmp.LOO < best.LOO
                best.LOO        = res_tmp.LOO;
                best.results    = res_tmp;
                best.ModelOpts  = Mo;
                best.poly       = poly_tmp;
                best.univ       = univ_tmp;
                best.ReduceModel= Red_tmp;
                best.choice     = struct('red_dim', rd, 'Order', p, 'qnorm', qn);
            end
        end
    end
end

%% ---------- 4) 写回最优模型 ----------
results                 = best.results;
ModelOpts               = best.ModelOpts;
ReduceModel             = best.ReduceModel;

results.poly_coeffs     = best.poly;
results.univ_vals       = best.univ;
results.Indices         = ModelOpts.Indices;

% 标准化参数（外层映射用）
results.mu              = mu;
results.sigma           = sigma;

% 统计信息
results.N_train         = size(data.X_train,1);
results.N_test          = ifelse(isfield(data,"X_val"), size(data.X_val,1), []);

% 可选：打印最终选择
fprintf('[aPC_Train] Best combo: red_dim=%s, Order=%d, qnorm=%g | LOO=%.4e\n', ...
    nan2dash(best.choice.red_dim), best.choice.Order, best.choice.qnorm, best.LOO);

end

%% ============ 辅助：若字段不存在给默认 ============
function v = getfield_or_default(S, fname, default)
    if isstruct(S) && isfield(S, fname) && ~isempty(S.(fname))
        v = S.(fname);
    else
        v = default;
    end
end

%% ============ 辅助：将标量/向量/空 转成行向量列表 ============
function L = to_list(x)
    if isempty(x)
        L = [];
    elseif isscalar(x)
        L = x;
    else
        L = x(:).';
    end
end

%% ============ 辅助：三目 ============
function o = ifelse(cond, a, b)
    if cond, o = a; else, o = b; end
end

%% ============ 辅助：NaN 打印为 '-' ============
function s = nan2dash(x)
    if isnan(x), s = '-'; else, s = num2str(x); end
end
