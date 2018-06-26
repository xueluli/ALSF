function [D0] = update_D0(Y, threshold)


if size(Y,1) >  size(Y,2)
    [U, S, V] = svd(Y, 'econ');
else
    [V, S, U] = svd(Y', 'econ');
end
th_S = wthresh(diag(S),'s',threshold);
D0 = U*diag(th_S)*V';


end

