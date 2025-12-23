%% High-dimensional test function (d = 300, Case 5)
function y = dim300(xx)

% xx: N x 300

x = xx;

% term 1: sum xi
term1 = sum(x, 2);

% term 2: 30*x1^2*x2^2
term2 = 30 .* (x(:,1).^2) .* (x(:,2).^2);

% term 3: sum_{i=2}^{300} xi^2 * x_{i-1}^2
term3 = sum(x(:,2:300).^2 .* x(:,1:299).^2, 2);

% term 4: nonlinear sinusoidal-exponential term
term4 = sum(sin(x) .* exp(x - 2), 2);

y = term1 + term2 + term3 - term4;

end
