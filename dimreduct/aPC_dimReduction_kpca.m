function [X_red, ReduceModel] = aPC_dimReduction_kpca(U, ModelOpts)
% aPC_dimReduction_kpca
% ---------------------------------------------------------
% 按 KPCA 官方 README 实现的降维适配器（无二次标准化）
%
% 训练降维： kpca.score         （README Example 02）
% 测试降维： results.score     （README Example 05）
%
% 不做二次标准化，由 aPC_dimReduction 统一处理
% ---------------------------------------------------------

    %% 1) 构造 kernel（若未指定则自动选择 gamma）
    if ~isfield(ModelOpts,'kernel') || isempty(ModelOpts.kernel)
        gamma = 1 / max(1, size(U,2));
        kernel = Kernel('type','gaussian','gamma',gamma);
    else
        kernel = ModelOpts.kernel;
    end

    %% 2) KPCA 参数
    parameter = struct('numComponents', ModelOpts.red_dim, ...
                       'kernelFunc', kernel);

    %% 3) 构建并训练 KPCA（Example 02）
    kpca = KernelPCA(parameter);
    kpca.train(U);

    %% 4) 训练降维结果 —— README 示例 02 明确规定：
    % mappingData = kpca.score
    X_red = kpca.score;   % ← 训练嵌入空间

    %% 5) 组装 ReduceModel（用于预测阶段 transform）
    ReduceModel = struct();
    ReduceModel.method = 'kpca';
    ReduceModel.kpca   = kpca;

    % transform：预测阶段必须使用 Example 05 方式
    ReduceModel.transform = @(U_new) project_new(U_new, kpca);

end


%% ========== 测试阶段降维（Example 05） ==========
function Z = project_new(U_new, kpca)
    % README Example 05:
    % results = kpca.test(testData)
    results = kpca.test(U_new);

    % 降维后的得分字段是 results.score
    Z = results.score;
end
