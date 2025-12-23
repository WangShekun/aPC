%% High-dimensional test function (d = 200)
function y = dim200(xx)

% xx: N x 200

x = xx;

term = x(:,1:199) + x(:,2:200).^2;
y = sum(term, 2) - 10;

end
