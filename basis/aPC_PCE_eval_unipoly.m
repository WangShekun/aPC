function [univ_vals, poly_coeffs] = aPC_PCE_eval_unipoly(U, MaxOrder, ModelOpts)
    BP = struct();
    BP.MaxDegrees = MaxOrder;  % 标量

    % 返回：
    %   univ_vals   : N × M × (P+1)，第三维对应 0..P
    %   poly_coeffs : 1×M cell，每个是 1×(P+1) cell，保存各阶多项式系数
    [univ_vals, poly_coeffs] = aPC_eval_univ_basis(U, BP, ModelOpts);
end