function InputOpts = dim100_Input()
for ii = 1:100
    InputOpts.Marginals(ii).Type = 'Uniform';
    if ii == 20
        InputOpts.Marginals(ii).Parameters = [1 3];
    else
        InputOpts.Marginals(ii).Parameters = [1 2];
    end
end
end