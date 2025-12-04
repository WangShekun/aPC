function kernel_str = aPC_format_kernel_str(ModelOpts)
% aPC_format_kernel_str
% -------------------------------------------------------
% Format kernel hyperparameters for display.
% Works for all kernel types and partial parameter sets.
% -------------------------------------------------------

if ~isfield(ModelOpts,'kernel_type') || isempty(ModelOpts.kernel_type)
    kernel_str = 'kernel=-';
    return;
end

ktype = ModelOpts.kernel_type;

params = {};

if isfield(ModelOpts,'kernel_gamma') && ~isempty(ModelOpts.kernel_gamma)
    params{end+1} = sprintf('γ=%g', ModelOpts.kernel_gamma);
end
if isfield(ModelOpts,'kernel_degree') && ~isempty(ModelOpts.kernel_degree)
    params{end+1} = sprintf('d=%g', ModelOpts.kernel_degree);
end
if isfield(ModelOpts,'kernel_offset') && ~isempty(ModelOpts.kernel_offset)
    params{end+1} = sprintf('c=%g', ModelOpts.kernel_offset);
end

if isempty(params)
    kernel_str = sprintf('kernel=%s', ktype);
else
    kernel_str = sprintf('kernel=%s(%s)', ktype, strjoin(params, ', '));
end

end
