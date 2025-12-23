function [X_red, ReduceModel] = aPC_dimReduction_mpldr(U, ModelOpts)
% aPC_dimReduction_mpldr
% ---------------------------------------------------------
%   MPLDR（Moment–Preserving Linear Dimensionality Reduction）
%   稳定版：Whitening + Adam + Stiefel 正交约束
%
%   输入：
%       U           - 上游标准化后的输入 (N × M)
%       ModelOpts   - 必须包含：
%                     .red_dim               → 降维维度 r
%                     .mpldr_opts (可选)     → 参数结构体
%
%   输出：
%       X_red       - 降维后训练特征 (N × r)
%       ReduceModel - transform() 用于预测
% ---------------------------------------------------------

%% =============== 读取降维维数 ===============
if ~isfield(ModelOpts, "red_dim")
    error("MPLDR 需要 ModelOpts.red_dim 指定降维维度");
end
r = ModelOpts.red_dim;

[N, d] = size(U);

%% =============== 读取/补全参数 ===============

if isfield(ModelOpts, "mpldr_opts")
    opts = ModelOpts.mpldr_opts;
else
    opts = struct();
end

% 默认参数
if ~isfield(opts,'maxIter'), opts.maxIter = 10000; end
if ~isfield(opts,'lr'),      opts.lr = 0.01; end
if ~isfield(opts,'alpha'),   opts.alpha = 1; end
if ~isfield(opts,'beta'),    opts.beta = 1; end
if ~isfield(opts,'gamma'),   opts.gamma = 1; end
if ~isfield(opts,'clip'),    opts.clip = 5; end

%% =============== Whitening: U_w = U * C^{-1/2} ===============
C = cov(U);
[V,D] = eig(C);
D_inv_sqrt = diag(1 ./ sqrt(diag(D) + 1e-12));
Wwhite = V * D_inv_sqrt * V';     % whitening matrix

U_w = U * Wwhite;

%% =============== 初始化 W（随机 + PCA 混合） ===============
[Vp, ~] = eigs(cov(U_w), r);
W = 0.8 * Vp + 0.2 * orth(randn(d, r));
[W, ~] = qr(W, 0);

%% =============== Adam 初始化 ===============
m = zeros(size(W));
v = zeros(size(W));
beta1 = 0.9;
beta2 = 0.999;
eps = 1e-8;

alpha = opts.alpha;
beta  = opts.beta;
gamma = opts.gamma;

%% =============== 主训练循环 ===============
loss_hist = zeros(opts.maxIter, 1);

for it = 1:opts.maxIter

    %% ---- Forward ----
    Z = U_w * W;   % N×r

    C_z = cov(Z);
    L2 = norm(C_z - eye(r), 'fro')^2;     % 二阶矩接近单位阵
    g3 = mean(Z.^3, 1);
    g4 = mean(Z.^4, 1);
    L3 = sum(g3.^2);
    L4 = sum((g4 - 3).^2);

    loss = alpha*L2 + beta*L3 + gamma*L4;
    loss_hist(it) = loss;

    %% ---- Backward dZ ----
    dL_dZ_2 = (4*alpha / N) * (Z * (C_z - eye(r)));
    dL_dZ_3 = (6*beta  / N) * (Z.^2 .* g3);
    dL_dZ_4 = (8*gamma / N) * (Z.^3 .* (g4 - 3));

    dL_dZ = dL_dZ_2 + dL_dZ_3 + dL_dZ_4;

    % ---- Gradient clipping ----
    dL_dZ = max(min(dL_dZ, opts.clip), -opts.clip);

    %% ---- Chain rule to W ----
    G = U_w' * dL_dZ;

    %% ---- Stiefel tangent projection ----
    Gtan = G - W * ((W' * G + G' * W) / 2);

    %% ---- Adam 更新 ----
    m = beta1*m + (1-beta1)*Gtan;
    v = beta2*v + (1-beta2)*(Gtan.^2);

    m_hat = m / (1 - beta1^it);
    v_hat = v / (1 - beta2^it);

    W_new = W - opts.lr * (m_hat ./ (sqrt(v_hat) + eps));

    %% ---- QR 保持正交 ----
    [W, ~] = qr(W_new, 0);
end

%% =============== 输出降维结果（回原空间） ===============
X_red = U * (Wwhite * W);

%% =============== ReduceModel 输出结构 ===============
ReduceModel.method = "mpldr";
ReduceModel.params.W       = W;
ReduceModel.params.Wwhite  = Wwhite;

ReduceModel.transform = @(U_new)( U_new * (Wwhite * W) );

ReduceModel.info.loss_hist = loss_hist;
ReduceModel.info.opts      = opts;

end
