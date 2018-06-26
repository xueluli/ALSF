function [pred] = ALSF_l1_pred(D,D0,Omega,Omega0,Y_test,K_train)

% addpath('utils');
C = size(D,2)/K_train;

D_range = K_train*(0:C);

X0 = Omega0*Y_test;
X  = Omega*Y_test;
Y = Y_test - D0*X0;


for i = 1:C
    Xi = get_block_row(X, i, D_range);
    Di = get_block_col(D, i, D_range);
    R = Y - Di*Xi;
    E(i,:) = sum(R.^2, 1);
end
[~, pred] = min(E);

end