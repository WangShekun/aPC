function aPC_print(results, ModelOpts, data, varargin)
% aPC_print - 可选择模块的 Arbitrary PCE 结果打印函数
%
% 可用模块选项：
%   "basic"          - 基本信息
%   "samples"        - 训练/测试信息
%   "errors"         - 误差指标
%   "coefficients"   - 多项式系数
%   "orthogonality"  - 正交性检查
%
% 特殊：
%   "all"            - 打印所有模块
%
% 默认（未指定时）打印：
%   basic + samples + errors

%% ---------- 默认模块 ----------
default_modules = ["basic", "samples", "errors", "orthogonality"];

if nargin < 4
    modules = default_modules;
else
    if any(strcmpi(varargin, "all"))
        modules = ["basic","samples","errors","coefficients","orth"];
    else
        modules = string(varargin);
    end
end
Indices = ModelOpts.Indices;
%% ---------- 打印框架 ----------
line        = repmat('-', 1, 78);
line_double = repmat('=', 1, 78);

fprintf('\n%s\n', line_double);
fprintf('Arbitrary Polynomial Chaos Expansion Report\n');
fprintf('%s\n', line);

%% ============================================================
%% 1) 基本信息（basic）
%% ============================================================
if any(modules == "basic")
    fprintf('%-28s : %s\n', 'Regression method', upper(results.method));
    fprintf('%-28s : %d\n', 'Total PCE terms', size(results.coefficients,1));
    fprintf('%-28s : %d\n', 'Max polynomial order', ModelOpts.Order);
    fprintf('%-28s : %.2f\n', 'q-norm', ModelOpts.qnorm);

    if strcmpi(results.method, 'LARS')
        fprintf('%-28s : %d\n', 'Active terms', nnz(results.coefficients));
    end

    %% ----- 降维信息打印（若存在） -----
    if isfield(ModelOpts,'ReductionMethod') && ~isempty(ModelOpts.ReductionMethod)
        fprintf('%-28s : %s\n', 'Dimension reduction', upper(ModelOpts.ReductionMethod));

        if isfield(ModelOpts,'red_dim') && ~isempty(ModelOpts.red_dim)
            fprintf('%-28s : %d\n', 'Reduced dimension', ModelOpts.red_dim);
        end

        % ---- KPCA 专属 kernel 信息 ----
        if strcmpi(ModelOpts.ReductionMethod,'kpca')
            kernel_str = aPC_format_kernel_str(ModelOpts);
            fprintf('%-28s : %s\n', 'KPCA kernel', kernel_str);
        end
    end


    fprintf('%s\n', line);
end

%% ============================================================
%% 2) 样本信息（samples）
%% ============================================================
if any(modules == "samples")
    if isfield(results,'N_train')
        fprintf('%-28s : %d\n', 'Training samples', results.N_train);
    else
        fprintf('%-28s : %s\n', 'Training samples', '—');
    end

    if isfield(results,'N_test')
        fprintf('%-28s : %d\n', 'Testing samples', results.N_test);
    else
        fprintf('%-28s : %s\n', 'Testing samples', '—');
    end

    fprintf('%-28s : %d\n', 'Input dimension', ModelOpts.num_vars);

    fprintf('%s\n', line);
end

%% ============================================================
%% 3) 误差指标（errors）
%% ============================================================
if any(modules == "errors")
    if isfield(results,'optErrorParams') && isfield(results.optErrorParams,'loo')
        fprintf('%-28s : %.6e\n', 'LOO error', results.optErrorParams.loo);
    else
        fprintf('%-28s : %s\n', 'LOO error', '—');
    end

    if isfield(results,'optErrorParams') && isfield(results.optErrorParams,'ModifiedLoo')
        fprintf('%-28s : %.6e\n', 'Modified LOO', results.optErrorParams.ModifiedLoo);
    else
        fprintf('%-28s : %s\n', 'Modified LOO', '—');
    end

    if isfield(results,'RelMSE')
        fprintf('%-28s : %.6e\n', 'RelMSE', results.RelMSE);
    else
        fprintf('%-28s : %s\n', 'RelMSE', '—');
    end

    fprintf('%s\n', line);
end

%% ============================================================
%% 4) 正交性检查（orth / orthogonality）
%% ============================================================
if any(modules == "orthogonality")

    fprintf('Orthogonality Check (per dimension):\n');

    X_train = data.X_train;

    U = results.map_U.transform(X_train);

    M = size(U, 2);
    Order = ModelOpts.Order;

    orthErr = zeros(1, M);

    for j = 1:M
        Uj = U(:, j);
        [psi_j, ~] = aPC_eval_aPC_2012(Order, Uj);
        orthErr(j) = aPC_check_orthogonality(psi_j);

        fprintf('  Dimension %d orth. error : %.3e\n', j, orthErr(j));
    end

    fprintf('  Max orthogonality error  : %.3e\n', max(orthErr));
    fprintf('%s\n', line);
end

%% ============================================================
%% 5) 系数（coefficients）
%% ============================================================
if any(modules == "coefficients")
    fprintf('PCE Coefficients:\n');
    fprintf('%-6s %-20s %s\n', 'Idx', 'Multi-Index', 'Coefficient');

    num_terms = length(results.coefficients);

    for k = 1:num_terms
        idx = sprintf('%d,', Indices(k,:));
        idx = ['[', idx(1:end-1), ']'];
        fprintf('%6d %-20s % .8e\n', k, idx, results.coefficients(k));
    end

    fprintf('%s\n', line);
end

%% ---------- 结束：总是打印双横线 ----------
fprintf('%s\n', line_double);

end
