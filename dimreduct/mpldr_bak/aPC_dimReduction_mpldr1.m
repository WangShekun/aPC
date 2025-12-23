function [X_red, ReduceModel] = aPC_dimReduction_mpldr(U, ModelOpts)
% aPC_dimReduction_mpldr
% -------------------------------------------------------
% 无监督 MPLDR（1–4 阶统计矩保持）降维模块
%
% 输入：
%   U          — 标准化后的输入数据 (N×M)
%   ModelOpts  — 必须包含:
%                  .red_dim            – 降维维度 r
%                  .mpldr_opts (可选) – MPLDR 参数结构体
%
% 输出：
%   X_red        — 降维后的训练特征 (N×r)
%   ReduceModel  — 预测阶段用 transform()
% -------------------------------------------------------

    if ~isfield(ModelOpts, 'red_dim')
        error('MPLDR 需要 ModelOpts.red_dim 指定降维维度。');
    end

    r = ModelOpts.red_dim;
    [N, d] = size(U);

    % ======= 读取 MPLDR 参数 =======
    if isfield(ModelOpts, 'mpldr_opts')
        opts = ModelOpts.mpldr_opts;
    else
        opts = struct();
    end

    % ======= 默认参数 =======
    if ~isfield(opts,'maxIter'),      opts.maxIter = 300; end
    if ~isfield(opts,'lr'),           opts.lr      = 0.01; end
    if ~isfield(opts,'alpha'),        opts.alpha   = 1; end
    if ~isfield(opts,'beta'),         opts.beta    = 1; end
    if ~isfield(opts,'gamma'),        opts.gamma   = 1; end
    if ~isfield(opts,'gamma_star'),   opts.gamma_star = 0; end
    if ~isfield(opts,'kappa_star'),   opts.kappa_star = 3; end

    % ======= 参数展开 =======
    alpha = opts.alpha;
    beta  = opts.beta;
    gamma = opts.gamma;

    gamma_star = opts.gamma_star;
    kappa_star = opts.kappa_star;

    maxIter = opts.maxIter;
    lr      = opts.lr;

    % ======= PCA 初始化 W =======
    C = cov(U);
    [V,~] = eigs(C, r, 'largestreal');
    W = V;

    loss_hist = zeros(maxIter,1);

    % ============================================================
    %                   MPLDR Optimizer (Stiefel)
    % ============================================================
    for it = 1:maxIter

        % ----- Forward -----
        Z = U * W;     % N × r

        C_Z = (Z' * Z) / N;
        E = C_Z - diag(diag(C_Z));

        gamma_vec = mean(Z.^3,1);
        kappa_vec = mean(Z.^4,1);

        L2 = alpha * norm(E, 'fro')^2;
        L3 = beta  * sum((gamma_vec - gamma_star).^2);
        L4 = gamma * sum((kappa_vec - kappa_star).^2);

        loss_hist(it) = L2 + L3 + L4;

        % ----- Backward dL/dZ -----
        dL_dZ_2 = (4*alpha/N) * (Z * E);
        dL_dZ_3 = (6*beta/N)  * (Z.^2 .* (gamma_vec - gamma_star));
        dL_dZ_4 = (8*gamma/N) * (Z.^3 .* (kappa_vec - kappa_star));

        dL_dZ = dL_dZ_2 + dL_dZ_3 + dL_dZ_4;

        % ----- chain rule: dL/dW -----
        G = U' * dL_dZ;

        % ----- Stiefel projection -----
        Gtan = G - W * ((W' * G + G' * W) / 2);

        % ----- gradient update -----
        W_new = W - lr * Gtan;

        % ----- QR retraction -----
        [Q,~] = qr(W_new, 0);
        W = Q;
    end

    % ========== 输出降维结果 ==========
    X_red = U * W;

    % ========== 构建 ReduceModel ==========
    ReduceModel.method = 'mpldr';
    ReduceModel.params.W = W;

    % transform 回调用于预测：U_new → Z_new
    ReduceModel.transform = @(U_new)( U_new * W );

    % 保存训练信息
    ReduceModel.info.loss_hist = loss_hist;
    ReduceModel.info.opts      = opts;
end
