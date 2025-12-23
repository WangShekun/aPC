function [X_red, ReduceModel] = aPC_dimReduction_mpldr(U, Y, ModelOpts)
% aPC_dimReduction_mpldr — MPLDR with PLS-style supervision (Version A)
% ---------------------------------------------------------
% 输入：
%   U         : (N×M) 已标准化输入
%   Y         : (N×1) 输出（可为空；lambda_y>0 且 Y 非空时启用监督）
%   ModelOpts : 需含 red_dim；可含 mpldr_opts.* 超参
%
% 目标：
%   L = alpha*||Cov(Z)-I||_F^2
%     + beta*sum(skew(Z).^2)
%     + gamma*sum((kurt(Z)-3).^2)
%     + lambda_y * ( - || Z' * Y ||_2^2 )     <-- PLS-style 强监督
%
% 数值技巧：Whitening + Adam + Cosine LR + 梯度裁剪 + Stiefel 约束(QR)
% 输出：
%   X_red       : (N×r) 训练期降维结果（未做二次标准化；外层统一处理）
%   ReduceModel : .transform(U_new) 预测映射；.params 含 Wwhite 与 W
% ---------------------------------------------------------

%% ====== 基本检查 ======
if ~isfield(ModelOpts,'red_dim') || isempty(ModelOpts.red_dim)
    error('MPLDR 需要在 ModelOpts.red_dim 中指定降维维数');
end
r = ModelOpts.red_dim;

[N, d] = size(U);

%% ====== 读取/补全超参（按你的 isfield 风格） ======
if ~isfield(ModelOpts,'mpldr_opts') || isempty(ModelOpts.mpldr_opts)
    opts = struct();
else
    opts = ModelOpts.mpldr_opts;
end

if ~isfield(opts,'maxIter')  || isempty(opts.maxIter),  opts.maxIter  = 300;  end
if ~isfield(opts,'lr')       || isempty(opts.lr),       opts.lr       = 0.01; end
if ~isfield(opts,'alpha')    || isempty(opts.alpha),    opts.alpha    = 1;    end
if ~isfield(opts,'beta')     || isempty(opts.beta),     opts.beta     = 1;    end
if ~isfield(opts,'gamma')    || isempty(opts.gamma),    opts.gamma    = 1;    end
if ~isfield(opts,'clip')     || isempty(opts.clip),     opts.clip     = 5;    end
if ~isfield(opts,'lambda_y') || isempty(opts.lambda_y), opts.lambda_y = 0;    end

maxIter  = opts.maxIter;
lr0      = opts.lr;
alpha    = opts.alpha;
beta_w   = opts.beta;
gamma_w  = opts.gamma;
clipv    = opts.clip;
lambda_y = opts.lambda_y;

%% ====== 监督开关（PLS-style，用未中心化 Y） ======
useY = false;
if lambda_y > 0 && ~isempty(Y)
    Y = Y(:);      % 不中心化、不缩放（PLS 风格）
    useY = true;
elseif lambda_y > 0
    warning('lambda_y > 0 但未提供 Y，自动退化为无监督');
    lambda_y = 0;
end

%% ====== Whitening: U_white = U * C^{-1/2} ======
C = cov(U);
[Vd, Dd] = eig(C);
D_inv_sqrt = diag(1 ./ sqrt(diag(Dd) + 1e-12));
Wwhite = Vd * D_inv_sqrt * Vd';
U_w = U * Wwhite;

%% ====== 初始化 W：PCA + 微扰 + QR ======
[Vp,~] = eigs(cov(U_w), r, 'largestreal');
W = Vp + 0.01*randn(d,r);
[W,~] = qr(W,0);

%% ====== Adam 初始化 ======
m = zeros(size(W));
v = zeros(size(W));
beta1=0.9; beta2=0.999; eps_adam=1e-8;

loss_hist = zeros(maxIter,1);

%% ===================== 训练循环 =======================
for it = 1:maxIter

    % —— Cosine 学习率衰减 ——
    lr = lr0 * 0.5 * (1 + cos(pi * it / maxIter));

    %% ---------- Forward ----------
    Z = U_w * W;                   % N×r

    % 二阶：白化到 I
    Cz = cov(Z);
    L2 = norm(Cz - eye(r),'fro')^2;

    % 三/四阶：偏度/峰度
    g3 = mean(Z.^3,1);
    g4 = mean(Z.^4,1);
    L3 = sum(g3.^2);
    L4 = sum((g4 - 3).^2);

    % PLS-style 监督：t = Z' * Y（不 /N，不中心化）
    Ly = 0;
    if useY
        t = Z' * Y;                % r×1
        Ly = - (t' * t);           % maximize ||t||
    end

    loss = alpha*L2 + beta_w*L3 + gamma_w*L4 + lambda_y*Ly;
    loss_hist(it) = loss;

    %% ---------- dL/dZ ----------
    dZ2 = (4*alpha/N) * (Z*(Cz - eye(r)));
    dZ3 = (6*beta_w/N) * (Z.^2 .* g3);
    dZ4 = (8*gamma_w/N) * (Z.^3 .* (g4 - 3));
    dL_dZ = dZ2 + dZ3 + dZ4;

    if useY
        % d/dZ of -(||Z'Y||^2) = -2 * Y * t'
        dLy_dZ = - 2 * (Y * t');   % N×r
        dL_dZ  = dL_dZ + lambda_y * dLy_dZ;
    end

    % —— 梯度裁剪 ——
    dL_dZ = max(min(dL_dZ, clipv), -clipv);

    % —— 梯度归一化（提升数值稳定性） ——
    gnorm = mean(sqrt(sum(dL_dZ.^2,2)));
    if gnorm > 1e-12
        dL_dZ = dL_dZ / gnorm;
    end

    %% ---------- 链式到 W ----------
    G = U_w' * dL_dZ;

    % Stiefel 切空间投影
    Gtan = G - W*((W'*G + G'*W)/2);

    %% ---------- Adam ----------
    m = beta1*m + (1-beta1)*Gtan;
    v = beta2*v + (1-beta2)*(Gtan.^2);
    m_hat = m / (1 - beta1^it);
    v_hat = v / (1 - beta2^it);

    W_new = W - lr * (m_hat ./ (sqrt(v_hat) + eps_adam));

    % 保持正交
    [W,~] = qr(W_new,0);
end

%% ===================== 输出与模型 =======================
X_red = U * (Wwhite * W);

ReduceModel.method = 'mpldr';
ReduceModel.params.W       = W;
ReduceModel.params.Wwhite  = Wwhite;

ReduceModel.transform = @(U_new)( U_new * (Wwhite * W) );

ReduceModel.info.loss_hist = loss_hist;
ReduceModel.info.opts = struct( ...
    'maxIter',maxIter,'lr',lr0,'alpha',alpha, ...
    'beta',beta_w,'gamma',gamma_w,'clip',clipv, ...
    'lambda_y',lambda_y,'supervision','PLS-style');
end
