function [results, ModelOpts, data, ReduceModel] = aPC_Train(data, ModelOpts)
% ------------------------------------------------------------
% 标准化 →（可选降维）→ 多索引生成 → 构造 Ψ → 回归 → LOO 选最优
%
% 支持自动搜索超参数（Order/qnorm/red_dim/kernel_* …）
% 搜索网格由 aPC_build_search_grid.m 自动生成
% ------------------------------------------------------------

%% ---------- 默认设置 ----------
if ~isfield(ModelOpts, 'RegressionMethod') || isempty(ModelOpts.RegressionMethod)
    ModelOpts.RegressionMethod = 'LARS';
end
if ~isfield(ModelOpts, 'qnorm') || isempty(ModelOpts.qnorm)
    ModelOpts.qnorm = 1;
end

%% ---------- 标准化 ----------
[data, map_U] = aPC_PCE_map_to_U(data.X_train, data);

%% ---------- 构造搜索网格 ----------
grid = aPC_build_search_grid(ModelOpts);
total = numel(grid);

if ~ModelOpts.Parallel
    %% ---------- 初始化进度条 ----------
    cnt = 0;
    bar_len = 40;
    fprintf('[aPC_Train] (Serial) Starting search over %d configurations...\n', total);

    best = struct('LOO', Inf);
    ReduceModel = [];

    for gi = 1:total

        cnt = cnt + 1;
        pct = cnt / total;

        % ---------- 进度条 ----------
        filled = round(pct * bar_len);
        bar_str = [repmat('#',1,filled), repmat('-',1,bar_len-filled)];

        % 当前候选参数（原始版本）
        Mo = grid(gi);

        % 打印降维信息
        if isfield(Mo,'red_dim') && ~isempty(Mo.red_dim)
            rd_str = num2str(Mo.red_dim);
        else
            rd_str = '-';
        end

        %% ---- 打印进度条信息 ----
        % 降维方法名（没有就显示 none）
        if isfield(Mo,'ReductionMethod') && ~isempty(Mo.ReductionMethod)
            redm = upper(Mo.ReductionMethod);
        else
            redm = 'NONE';
        end
        
        % KPCA 才显示 kernel 参数
        if isfield(Mo,'ReductionMethod') && strcmpi(Mo.ReductionMethod,'kpca')
            extra_str = [' | KPCA(' aPC_format_kernel_str(Mo) ')'];
        else
            extra_str = '';
        end
        
        fprintf('\r[aPC_Train] [%s] %3d%% \n| RD=%s | rd=%s | p=%d | q=%g%s |', ...
            bar_str, round(pct*100), redm, rd_str, Mo.Order, Mo.qnorm, extra_str);


        %% ---------- 降维 ----------
        if isfield(Mo,'red_dim') && ~isempty(Mo.red_dim)
            [data_tmp, Mo_tmp, Red_tmp] = aPC_dimReduction(data, Mo);
        else
            data_tmp = data;
            Mo_tmp = Mo;
            Red_tmp = [];
            if ~isfield(Mo_tmp,'num_vars') || isempty(Mo_tmp.num_vars)
                Mo_tmp.num_vars = size(data_tmp.U,2);
            end
        end

        % ---------- 多索引 ----------
        Mo_tmp.Indices = aPC_GenerateTerms(Mo_tmp);

        % ---------- 构造多维基函数 ----------
        [Psi_tmp, poly_tmp, univ_tmp] = aPC_PCE_build_Psi(Mo_tmp, data_tmp);

        % ---------- 回归 ----------
        res_tmp = aPC_RegressionMethods(Psi_tmp, data_tmp.Y_train, Mo_tmp.RegressionMethod);

        % ---------- 更新最优 ----------
        if res_tmp.LOO < best.LOO
            best.LOO        = res_tmp.LOO;
            best.results    = res_tmp;
            best.ModelOpts  = Mo_tmp;
            best.poly       = poly_tmp;
            best.univ       = univ_tmp;
            best.ReduceModel= Red_tmp;
            best.choice     = Mo;
        end

        % aPC_print(res_tmp, Mo_tmp, Mo_tmp.Indices);
    end

    fprintf('\n'); % 清理进度条

    %% ---------- 写回最终结果 ----------
    results             = best.results;
    ModelOpts           = best.ModelOpts;
    ReduceModel         = best.ReduceModel;

    results.poly_coeffs = best.poly;
    results.univ_vals   = best.univ;
    results.Indices     = ModelOpts.Indices;
    results.map_U       = map_U;   % 标准化映射

    results.N_train     = size(data.X_train,1);
    if isfield(data,"X_val")
        results.N_test  = size(data.X_val,1);
    else
        results.N_test  = [];
    end

    %% --------- 打印最佳模型 ---------
    mo = ModelOpts;
    
    % --- 降维方法名 ---
    if isfield(mo,'ReductionMethod') && ~isempty(mo.ReductionMethod)
        redm = upper(mo.ReductionMethod);
    else
        redm = 'NONE';
    end
    
    % --- 降维维度 ---
    if isfield(mo,'red_dim') && ~isempty(mo.red_dim)
        rd_str = num2str(mo.red_dim);
    else
        rd_str = 'none';
    end
    
    % --- kernel 信息（仅 KPCA） ---
    if strcmpi(redm,'KPCA')
        extra_str = [' | KPCA(' aPC_format_kernel_str(mo) ')'];
    else
        extra_str = '';
    end
    
    fprintf('[aPC_Train] Best Model: RD=%s | rd=%s | p=%d | q=%g%s | LOO=%e\n', ...
            redm, rd_str, mo.Order, mo.qnorm, extra_str, results.LOO);

    return;  % 串行版本结束
end

%% ============================================================
%                   并行版本（parallel = true）
% ============================================================
fprintf('[aPC_Train] (Parallel) Starting search over %d configurations...\n', total);

% —— 单行进度条 ——
dq = parallel.pool.DataQueue;
cnt = 0; bar_len = 40;
afterEach(dq, @updateBar);

    function updateBar(~)
        cnt = cnt + 1;
        pct = cnt / total;
        filled  = round(pct * bar_len);
        bar_str = [repmat('#',1,filled), repmat('-',1,bar_len-filled)];
        fprintf('\r[aPC_Train] [%s] %3d%%  (%d/%d)', ...
            bar_str, round(pct*100), cnt, total);
    end

% —— 开并行池 ——
if isempty(gcp('nocreate')), parpool; end

% —— 并行评估 -------
outs(total) = struct('LOO',Inf);

parfor gi = 1:total
    outs(gi) = aPC_evaluateCandidate_LOO(data, grid(gi));
    send(dq, gi);
end

fprintf('\n');

% —— 最优组合 ——
[~, best_idx] = min([outs.LOO]);
best_choice   = grid(best_idx);

% —— 串行重训（完整版本） ——
Mo = best_choice;

if isfield(Mo,'red_dim') && ~isempty(Mo.red_dim)
    [data_tmp, Mo_tmp, Red_tmp] = aPC_dimReduction(data, Mo);
else
    data_tmp = data; Mo_tmp = Mo; Red_tmp = [];
    if ~isfield(Mo_tmp,'num_vars') || isempty(Mo_tmp.num_vars)
        Mo_tmp.num_vars = size(data_tmp.U,2);
    end
end

Mo_tmp.Indices = aPC_GenerateTerms(Mo_tmp);
[Psi, poly_coeffs, univ_vals] = aPC_PCE_build_Psi(Mo_tmp, data_tmp);
res = aPC_RegressionMethods(Psi, data_tmp.Y_train, Mo_tmp.RegressionMethod);

% —— 写回最终结果 ——
results                 = res;
ModelOpts               = Mo_tmp;
ReduceModel             = Red_tmp;
results.poly_coeffs     = poly_coeffs;
results.univ_vals       = univ_vals;
results.Indices         = ModelOpts.Indices;
results.map_U           = map_U;

results.N_train         = size(data.X_train,1);

if isfield(data,"X_val")
    results.N_test = size(data.X_val,1);
else 
    results.N_test = [];
end

% —— 格式化 red_dim 与 kernel ——
if isfield(best_choice,'red_dim') && ~isempty(best_choice.red_dim)
    rd_str = num2str(best_choice.red_dim);
else
    rd_str = '-';
end
kernel_str = aPC_format_kernel_str(best_choice);

%% --------- 打印最佳模型 ---------
mo = ModelOpts;

% --- 降维方法名 ---
if isfield(mo,'ReductionMethod') && ~isempty(mo.ReductionMethod)
    redm = upper(mo.ReductionMethod);
else
    redm = 'NONE';
end

% --- 降维维度 ---
if isfield(mo,'red_dim') && ~isempty(mo.red_dim)
    rd_str = num2str(mo.red_dim);
else
    rd_str = 'none';
end

% --- kernel 信息（仅 KPCA） ---
if strcmpi(redm,'KPCA')
    extra_str = [' | KPCA(' aPC_format_kernel_str(mo) ')'];
else
    extra_str = '';
end

fprintf('[aPC_Train] Best Model: RD=%s | rd=%s | p=%d | q=%g%s | LOO=%e\n', ...
        redm, rd_str, mo.Order, mo.qnorm, extra_str, results.LOO);



end