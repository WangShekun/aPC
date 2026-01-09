function PCE_terms = aPC_GenerateTerms(ModelOpts)
    % PC_GenerateTerms：生成多项式阶数组合，并应用 q-norm 截断
    % InputOpts.Order：最大阶数
    % InputOpts.qnorm：q-norm参数
    % InputOpts.p：截断的阈值（默认为1，表示不进行截断）
    
    Order = ModelOpts.Order;      % 最大阶数
    qnorm = ModelOpts.qnorm;      % q-norm参数（默认为1，表示不进行截断）
    num_vars = ModelOpts.num_vars;  % 输入变量个数
    
    % 如果未指定qnorm，默认设置为1（不进行截断）
    if isempty(qnorm)
        qnorm = 1;  % 默认q-norm为1，表示不做截断
    end
    
    % 生成所有阶数组合
    PCE_terms = generate_combinations(Order, num_vars);
    
    % 如果q-norm小于1，则应用q-norm截断
    if qnorm < 1
        PCE_terms = apply_qnorm_truncation(PCE_terms, qnorm, Order);
    end
end

function combinations = generate_combinations(order, num_vars)
    % 递归生成阶数组合，支持任意数量的自变量
    if num_vars == 1
        combinations = (0:order)';  % 返回一个列向量
    else
        combinations = [];
        for i = 0:order
            sub_combinations = generate_combinations(order - i, num_vars - 1);
            % 将子组合和当前i拼接成一个新的矩阵
            combinations = [combinations; [repmat(i, size(sub_combinations, 1), 1), sub_combinations]];
        end
    end
end

function filtered_combinations = apply_qnorm_truncation(combinations, qnorm, max_order)
    % 根据 qnorm 截断筛选阶数组合，并确保阶数不超过最大阶数
    norms = zeros(size(combinations, 1), 1);  % 存储每个阶数组合的 q-norm
    for i = 1:size(combinations, 1)
        alpha = combinations(i, :);
        norms(i) = compute_qnorm(alpha, qnorm);  % 计算 q-norm
    end
    
    % 仅保留 q-norm 小于等于阶数且总阶数不超过最大阶数的项
    valid_combinations = norms <= max_order & sum(combinations, 2) <= max_order;  % 阶数之和不超过最大阶数
    filtered_combinations = combinations(valid_combinations, :);  % 保留符合条件的阶数组合
end

function qnorm_value = compute_qnorm(alpha, qnorm)
    % 根据阶数组合计算 q-norm
    qnorm_value = (sum(abs(alpha).^qnorm))^(1/qnorm);  % q-norm计算公式
end