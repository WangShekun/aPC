function [X_red, ReduceModel] = aPC_dimReduction_mpldr(U, Y, ModelOpts)
% aPC_dimReduction_mpldr — 完整加强版 MPLDR（线性降维，2–4阶矩保持 + 可选弱监督）
% -----------------------------------------------------------------------------
% 输入:
%   U         : (N×M) 已标准化输入（均值0、方差1）
%   Y         : (N×1) 输出（可为空；若启用监督且 Y 为空则自动退化为无监督）
%   ModelOpts : 必须包含 red_dim；可在 ModelOpts.mpldr_opts 中设置超参
%
% 目标（默认）:
%   L = alpha * ||Cov(Z) - I||_F^2
%     + beta  * sum_i ( skew_i - skew_target )^2
%     + gamma * sum_i ( kurt_i - kurt_target )^2
%     + lambda_y * L_y(Z, Y)                                 (监督项可选)
%
% 监督项:
%   supervision_mode = 'pls' (默认) : L_y = - || Z' * Y ||_2^2
%   supervision_mode = 'corr'       : L_y = - || (Z' * (Y-meanY)) / N ||_2^2
%
% 数值策略:
%   - Whitening 开关（默认开启）
%   - Adam 优化 + 余弦学习率衰减
%   - 梯度裁剪 + 梯度归一化
%   - 课程式训练：前 warmup_frac 轮更强调监督（lambda_y 线性升权）
%   - 可选 mini-batch（默认全量）
%   - Stiefel 正交约束（QR 回缩）
%
% 返回:
%   X_red        : (N×r) 降维训练特征（未做二次标准化；由上层统一处理）
%   ReduceModel  : .transform(U_new) 预测映射；.params.Wwhite 与 .params.W
% -----------------------------------------------------------------------------

%% ===== 基本检查 =====
if ~isfield(ModelOpts,'red_dim') || isempty(ModelOpts.red_dim)
    error('MPLDR 需要在 ModelOpts.red_dim 中指定降维维数');
end
r = ModelOpts.red_dim;

[N, d] = size(U);

%% ===== 读取/补默认超参（仅用 isfield / isempty）=====
if ~isfield(ModelOpts,'mpldr_opts') || isempty(ModelOpts.mpldr_opts)
    opts = struct();
else
    opts = ModelOpts.mpldr_opts;
end

% 训练与优化
if ~isfield(opts,'maxIter')  || isempty(opts.maxIter),  opts.maxIter  = 400; end
if ~isfield(opts,'lr')       || isempty(opts.lr),       opts.lr       = 0.01; end
if ~isfield(opts,'clip')     || isempty(opts.clip),     opts.clip     = 5;    end
if ~isfield(opts,'batch_size') || isempty(opts.batch_size), opts.batch_size = 0; end  % 0 = full batch

% 目标权重与目标值
if ~isfield(opts,'alpha')    || isempty(opts.alpha),    opts.alpha    = 1;    end
if ~isfield(opts,'beta')     || isempty(opts.beta),     opts.beta     = 1;    end
if ~isfield(opts,'gamma')    || isempty(opts.gamma),    opts.gamma    = 1;    end
if ~isfield(opts,'skew_target')  || isempty(opts.skew_target),  opts.skew_target  = 0; end
if ~isfield(opts,'kurt_target')  || isempty(opts.kurt_target),  opts.kurt_target  = 3; end

% 监督项
if ~isfield(opts,'lambda_y') || isempty(opts.lambda_y), opts.lambda_y = 0;    end
if ~isfield(opts,'supervision_mode') || isempty(opts.supervision_mode), opts.supervision_mode = 'pls'; end
if ~isfield(opts,'lr_y_mult') || isempty(opts.lr_y_mult), opts.lr_y_mult = 2.0; end   % 监督梯度放大系数

% 课程式训练（前 warmup_frac 轮侧重监督）
if ~isfield(opts,'warmup_frac') || isempty(opts.warmup_frac), opts.warmup_frac = 0.2; end
opts.warmup_frac = max(0,min(0.9,opts.warmup_frac));

% 白化开关
if ~isfield(opts,'whiten') || isempty(opts.whiten), opts.whiten = 1; end

% 固定随机种子（可选）
if ~isfield(opts,'seed') || isempty(opts.seed), opts.seed = []; end
if ~isempty(opts.seed), rng(opts.seed); end

maxIter   = opts.maxIter;
lr0       = opts.lr;
clipv     = opts.clip;
batch_sz  = opts.batch_size;

alpha     = opts.alpha;
beta_w    = opts.beta;
gamma_w   = opts.gamma;
skew_t    = opts.skew_target;
kurt_t    = opts.kurt_target;

lambda_y  = opts.lambda_y;
mode_y    = lower(opts.supervision_mode);
lr_y_mult = opts.lr_y_mult;
warmup_f  = opts.warmup_frac;
useY      = (lambda_y > 0) && ~isempty(Y);

if useY
    Y = Y(:);
    if strcmp(mode_y,'corr')
        y_c = Y - mean(Y);
    end
end

%% ===== Whitening（可开关）=====
if opts.whiten
    C = cov(U);
    [Vd, Dd] = eig(C);
    D_inv_sqrt = diag(1 ./ sqrt(max(diag(Dd),0) + 1e-12));
    Wwhite = Vd * D_inv_sqrt * Vd';
else
    Wwhite = eye(d);
end
U_w = U * Wwhite;

%% ===== 初始化 W：PCA + 微扰 + QR =====
[Vp,~] = eigs(cov(U_w), r, 'largestreal');
W = Vp + 0.01*randn(d,r);
[W,~] = qr(W,0);

%% ===== Adam 初始化 =====
m = zeros(size(W)); v = zeros(size(W));
beta1=0.9; beta2=0.999; eps_adam=1e-8;

loss_hist = zeros(maxIter,1);

%% ===== Mini-batch 索引（如启用）=====
if batch_sz<=0 || batch_sz>=N
    use_batch = false;
else
    use_batch = true;
end

%% ===================== 训练循环 =======================
for it = 1:maxIter

    % 余弦学习率衰减
    lr = lr0 * 0.5 * (1 + cos(pi * it / maxIter));

    % 课程式监督权重调度（前 warmup_frac 轮线性升温）
    if useY
        if it <= floor(warmup_f * maxIter)
            sched = it / max(1, floor(warmup_f * maxIter));
        else
            sched = 1.0;
        end
        lambda_eff = lambda_y * sched;   % 生效监督强度
    else
        lambda_eff = 0;
    end

    % ------- 取 batch -------
    if use_batch
        idx = randsample(N, batch_sz);
        Uw_b = U_w(idx, :);
        if useY
            Y_b = Y(idx);
            if strcmp(mode_y,'corr'), y_c_b = y_c(idx); end
        end
        Nb = batch_sz;
    else
        Uw_b = U_w;
        Nb   = N;
        if useY
            Y_b = Y;
            if strcmp(mode_y,'corr'), y_c_b = y_c; end
        end
    end

    %% ---------- Forward ----------
    Z = Uw_b * W;                  % Nb×r

    % 二阶：白化到 I
    Cz = (Z' * Z) / Nb;            % 更稳的 batch 协方差
    L2 = norm(Cz - eye(r),'fro')^2;

    % 三/四阶：偏度/峰度（batch 估计）
    g3 = mean(Z.^3,1);
    g4 = mean(Z.^4,1);
    L3 = sum((g3 - skew_t).^2);
    L4 = sum((g4 - kurt_t).^2);

    % 监督项
    Ly = 0;
    if lambda_eff > 0
        if strcmp(mode_y,'pls')
            t = Z' * Y_b;            % r×1
            Ly = - (t' * t);         % maximize ||t||
        else % 'corr'
            t = (Z' * y_c_b) / Nb;   % r×1
            Ly = - (t' * t);
        end
    end

    loss = alpha*L2 + beta_w*L3 + gamma_w*L4 + lambda_eff*Ly;
    loss_hist(it) = loss;

    %% ---------- dL/dZ ----------
    dZ2 = (4*alpha/Nb) * (Z * (Cz - eye(r)));
    dZ3 = (6*beta_w/Nb) * (Z.^2 .* (g3 - skew_t));
    dZ4 = (8*gamma_w/Nb) * (Z.^3 .* (g4 - kurt_t));
    dL_dZ = dZ2 + dZ3 + dZ4;

    if lambda_eff > 0
        if strcmp(mode_y,'pls')
            % d/dZ of -(||Z'Y||^2) = -2 * Y * t'
            dLy_dZ = - 2 * (Y_b * t');
        else % 'corr' : -(|| (Z'^y_c)/N ||^2) → -(2/N) y_c t'
            dLy_dZ = - (2/Nb) * (y_c_b * t');
        end
        dL_dZ = dL_dZ + (lr_y_mult * lambda_eff) * dLy_dZ;
    end

    % 梯度裁剪
    dL_dZ = max(min(dL_dZ, clipv), -clipv);

    % 梯度归一化（提升稳定性）
    gnorm = mean( sqrt(sum(dL_dZ.^2,2)) );
    if gnorm > 1e-12
        dL_dZ = dL_dZ / gnorm;
    end

    %% ---------- 链式到 W ----------
    G = Uw_b' * dL_dZ;

    % 投影到 Stiefel 切空间
    Gtan = G - W*((W'*G + G'*W)/2);

    %% ---------- Adam ----------
    m = beta1*m + (1-beta1)*Gtan;
    v = beta2*v + (1-beta2)*(Gtan.^2);
    m_hat = m / (1 - beta1^it);
    v_hat = v / (1 - beta2^it);

    W_new = W - lr * (m_hat ./ (sqrt(v_hat) + eps_adam));

    % QR 回缩保证列正交
    [W,~] = qr(W_new,0);
end

%% ===================== 输出与模型 =======================
X_red = U * (Wwhite * W);

ReduceModel.method = 'mpldr';
ReduceModel.params.W       = W;
ReduceModel.params.Wwhite  = Wwhite;

ReduceModel.transform = @(U_new)( U_new * (Wwhite * W) );

ReduceModel.info.loss_hist = loss_hist;
ReduceModel.info.opts = opts;
ReduceModel.info.details = struct( ...
    'used_supervision', useY, ...
    'supervision_mode', mode_y, ...
    'lambda_effective_final', lambda_eff, ...
    'whitening', logical(opts.whiten), ...
    'batch_size', batch_sz);

end
