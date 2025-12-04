function grid = aPC_build_search_grid(ModelOpts)
% aPC_build_search_grid
% -------------------------------------------------------
% Build the hyperparameter grid automatically:
% - scalar value → fixed
% - vector/cell  → participate in grid search
% Only fields listed in aPC_list_search_fields() are considered.
% -------------------------------------------------------

search_fields = aPC_list_search_fields();

keynames = {};
keyvals  = {};

for i = 1:numel(search_fields)
    fn = search_fields{i};

    if ~isfield(ModelOpts, fn)
        continue;
    end

    val = ModelOpts.(fn);

    % scalar → ignore
    if isnumeric(val) && isscalar(val)
        continue;
    end

    % vector numeric → convert to cell list
    if isnumeric(val)
        keynames{end+1} = fn;
        keyvals{end+1}  = num2cell(val(:)');
        continue;
    end

    % cell list → direct search space
    if iscell(val)
        keynames{end+1} = fn;
        keyvals{end+1}  = val(:)';
        continue;
    end
end

% no search → return single model
if isempty(keynames)
    grid = ModelOpts;
    return;
end

lens = cellfun(@numel,keyvals);
idx  = ones(1,numel(keynames));
configs = {};

while true
    Mo = ModelOpts;
    for k = 1:numel(keynames)
        Mo.(keynames{k}) = keyvals{k}{idx(k)};
    end
    configs{end+1} = Mo;

    % increment
    j = numel(idx);
    while j>=1
        idx(j)=idx(j)+1;
        if idx(j) <= lens(j)
            break;
        end
        idx(j)=1;
        j=j-1;
    end
    if j <= 0, break; end
end

grid = [configs{:}];

end
