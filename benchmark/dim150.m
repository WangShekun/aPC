%% High-dimensional test function (d = 150, Case 1)
function y = dim150(xx)

% xx: N x 150

x = xx;
N = size(x,1);

% term 1: sum xi
term1 = sum(x, 2);

% term 2: 30*x1*x2
term2 = 30 .* x(:,1) .* x(:,2);

% term 3: sum_{i=2}^{149} xi * x_{i+1}
term3 = sum(x(:,2:149) .* x(:,3:150), 2);

% term 4: nonlinear sinusoidal-exponential term
term4 = sum(sin(x) .* exp(x - 2), 2);

y = term1 + term2 + term3 + term4 - 10;

end
