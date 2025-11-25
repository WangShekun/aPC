function [data, ModelOpts, ReduceModel] = aPC_dimReduction(data, ModelOpts)
% 通用降维接口，当前方法=PLS，将来可扩展 PCA、ICA、KPCA…

    % 若没有降维要求
    if ~isfield(ModelOpts,'ReductionMethod') || isempty(ModelOpts.ReductionMethod)
        ReduceModel = [];
        return;
    end

    method = lower(ModelOpts.ReductionMethod);
    U = data.U;              % ← 已标准化过的输入
    Y = data.Y_train;

    switch method
        % ============================================
        %                 P L S
        % ============================================
        case 'pls'
            ncomp = ModelOpts.red_dim;

            % Intercept=false 是正确的，因为 U 已经标准化了
            [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = ...
                plsregress(U, Y, ncomp, 'Intercept', false);

            % ------- transform 所需的最小信息 -------
            params.W = stats.W;        % 正向降维矩阵

            ReduceModel.method    = 'pls';
            ReduceModel.params    = params;

            % ------- 通用化 transform -------
            ReduceModel.transform = @(U_new)( U_new * params.W );

            % ------- 保存全部信息以备未来扩展 -------
            info.XL     = XL;
            info.YL     = YL;
            info.XS     = XS;
            info.YS     = YS;
            info.BETA   = BETA;
            info.PCTVAR = PCTVAR;
            info.MSE    = MSE;
            info.stats  = stats;   % T2, Residuals…

            ReduceModel.info = info;

            % 降维后的训练数据
            X_red = XS;

        % ============================================
        %           未来方法（预留示例）
        % ============================================
        case 'pca'
            ncomp = ModelOpts.red_dim;

            [coeff, score, ~] = pca(U);

            params.V = coeff(:,1:ncomp);

            ReduceModel.method = 'pca';
            ReduceModel.params = params;

            ReduceModel.transform = @(U_new)( U_new * params.V );

            info.coeff = coeff;
            info.score = score;

            ReduceModel.info = info;

            X_red = score(:,1:ncomp);

        otherwise
            error('Unknown reduction method: %s', method);
    end

    % 更新数据和维度
    data.U = X_red;
    ModelOpts.num_vars = size(X_red,2);
end
