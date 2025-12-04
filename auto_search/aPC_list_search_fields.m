function fields = aPC_list_search_fields()
% aPC_list_search_fields
% -------------------------------------------------------
% List of fields that support scalar-or-vector auto-search.
% Add new fields here when new hyperparameters are introduced.
% -------------------------------------------------------

fields = {
    'Order'          % polynomial order
    'qnorm'          % q-norm truncation
    'red_dim'        % dimensionality reduction target size

    % KPCA kernel hyperparameters
    'kernel_type'
    'kernel_gamma'
    'kernel_degree'
    'kernel_offset'

    % Future extension:
    % 'ridge_lambda'
    % 'lasso_alpha'
    % 'poly_trunc_level'
};

end
