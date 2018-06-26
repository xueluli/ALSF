function [XX,X,X0,xi,cost] = XX0_update(XX,D,D0,Y,Omega,Omega0,C,N_train,L,Ki,optX)
% addpath('utils');
% addpath('../utils/FISTA-master');

lambda2 = optX.lambda2;
lambda3 = optX.lambda3;
lambda4 = optX.lambda4;
tau = optX.tau;
K0 = optX.K0;

ind1 = size(D{1},2); % the colum number of P1
ind2 = size(Y{1},2); % the colum number of Z1

% GX = zeros(ind1*C,ind2*C); % the gradient of X
% GX0 = zeros(ind1*C,ind2*C); % the gradient of X_0

    function [X,X0,xi,xij] = extractFromXX(XX)
        for i = 1:C
            p1 = (i-1)*N_train;
            p2 = (i-1)*Ki;
            X{i} = XX(1:Ki*C,p1+1:p1+N_train);
            X0{i} = XX(Ki*C+1:end,p1+1:p1+N_train);
            xi{i} = X{i}(p2+1:p2+Ki,:); % extract Xcc
            for j = 1:C
                xij{(i-1)*C+j} = XX(p2+1:p2+Ki,(j-1)*N_train+1:j*N_train);  % extract Xc\bar(c)
            end
        end
    end

    function gX = grad(XX)
        
        [X,X0,xi,xij] = extractFromXX(XX);
        
        YZ2 = []; % for calculate X0
        X00 = []; % for calculate X0
        for i = 1:C
            Z1{i} = [Y{i}-D0*X0{i}; sqrt(tau*lambda2)*Omega{i}*Y{i}];
%             size(D{i})
%             size(sqrt(tau*lambda2)*eye(N_train))
            P1{i} = [D{i};sqrt(tau*lambda2)*eye(Ki)];
    
            for j = 1:C
                if i == j
                    
                    GX((i-1)*ind1+1:i*ind1,(i-1)*ind2+1:i*ind2) = -2*P1{i}'*Z1{i}+2*P1{i}'*P1{i}*xi{i};
                    
                else
                    
                    GX((i-1)*ind1+1:i*ind1,(j-1)*ind2+1:j*ind2) = tau*lambda4*(-2*Omega{i}*Y{j}+2*xij{(i-1)*C+j});
                end
            end
            Z2 = [Y{i}-D{i}*xi{i}; sqrt(tau*lambda3)*Omega0*Y{i}];
            YZ2 = [YZ2 Z2];
            X00 = [X00 X0{i}];
        end
        
        P2 = [D0; sqrt(tau*lambda3)*eye(K0)];
        GX0 = -2*P2'*YZ2+2*P2'*P2*X00;
        gX = [GX;GX0];
%         /Ki; % the gradient of X_total
%         gX = 2*[GX;GX0]/Ki/3; % the gradient of X_total
        
    end

    function cost = calc_F(XX)
        [X,X0,xi,xij] = extractFromXX(XX);
        cost = 0;
        for c = 1:C
            a = normF2(Y{c}-D{c}*xi{c}-D0*X0{c});
            b = tau*lambda2*normF2(xi{c}-Omega{c}*Y{c});
            d = tau*lambda3*normF2(X0{c}-Omega0*Y{c});
            Ycnot = [];
            Xcnot = [];
            for j = 1:C
                if j == c
                    Ycnot = Ycnot;
                    Xcnot = Xcnot;
                else
                    Ycnot = [Ycnot Y{j}];
                    Xcnot = [Xcnot xij{(c-1)*C+j}];
                end
            end
            e = tau*lambda4*normF2(Xcnot-Omega{c}*Ycnot);
            cost = (cost+a+b+d+e)+0.1*sum(sum(abs(XX)));
        end
        
    end

function cost = calc_f(XX)
        [X,X0,xi,xij] = extractFromXX(XX);
        cost = 0;
        for c = 1:C
            a = normF2(Y{c}-D{c}*xi{c}-D0*X0{c})/Ki;
            b = tau*lambda2*normF2(xi{c}-Omega{c}*Y{c})/Ki;
            d = tau*lambda3*normF2(X0{c}-Omega0*Y{c})/Ki;
            Ycnot = [];
            Xcnot = [];
            for j = 1:C
                if j == c
                    Ycnot = Ycnot;
                    Xcnot = Xcnot;
                else
                    Ycnot = [Ycnot Y{j}];
                    Xcnot = [Xcnot xij{(c-1)*C+j}];
                end
            end
            e = tau*lambda4*normF2(Xcnot-Omega{c}*Ycnot)/Ki;
            cost = cost+a+b+d+e;
        end
        
end

function [X, iter] = fista(grad, Xinit, L, lambda, opts, calc_F)   
% function [X, iter] = fista(grad, Xinit, L, lambda, opts, calc_F)   
% * A Fast Iterative Shrinkage-Thresholding Algorithm for 
% Linear Inverse Problems.
% * Solve the problem: `X = arg min_X F(X) = f(X) + lambda||X||_1` where:
%   - `X`: variable, can be a matrix.
%   - `f(X)` is a smooth convex function with continuously differentiable 
%       with Lipschitz continuous gradient `L(f)` (Lipschitz constant of 
%       the gradient of `f`).
% * Syntax: `[X, iter] = FISTA(calc_F, grad, Xinit, L, lambda,  opts)` where:
%   - INPUT:
%     + `grad`: a _function_ calculating gradient of `f(X)` given `X`.
%     + `Xinit`: initial guess.
%     + `L`: the Lipschitz constant of the gradient of `f(X)`.
%     + `lambda`: a regularization parameter, can be either a scalar or 
%       a weighted matrix.
%     + `opts`: a _structure_ variable describing the algorithm.
%       * `opts.max_iter`: maximum iterations of the algorithm. 
%           Default `300`.
%       * `opts.tol`: a tolerance, the algorithm will stop if difference 
%           between two successive `X` is smaller than this value. 
%           Default `1e-8`.
%       * `opts.show_progress`: showing `F(X)` after each iteration or not. 
%           Default `false`. 
%     + `calc_F`: optional, a _function_ calculating value of `F` at `X` 
%       via `feval(calc_F, X)`. 
% -------------------------------------
% Author: Tiep Vu, thv102, 4/6/2016
% (http://www.personal.psu.edu/thv102/)
% -------------------------------------
    opts = initOpts(opts);
    Linv = 1/L;    
    lambdaLiv = lambda*Linv;
    x_old = Xinit;
    y_old = Xinit;
    t_old = 1;
    iter = 0;
    cost_old = 1e10;
    %% MAIN LOOP
    while  iter < opts.max_iter
%         iter
        iter = iter + 1;
        x_new = shrinkage(y_old - Linv*feval(grad, y_old), lambdaLiv);
%         x_new
        t_new = 0.5*(1 + sqrt(1 + 4*t_old^2));
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old);
        %% check stop criteria
        e = norm1(x_new - x_old)/numel(x_new);
        if e < opts.tol
            break;
        end
        %% update
        x_old = x_new;
        t_old = t_new;
        y_old = y_new;
        %% show progress
        if opts.verbose
            if nargin ~= 0
                cost_new = feval(calc_F, x_new);
%                 if cost_new <= cost_old 
%                     stt = 'YES.';
%                 else 
%                     stt = 'NO, check your code.';
%                 end
%                 fprintf('iter = %3d, cost = %f, cost decreases? %s\n', ...
%                     iter, cost_new, stt);
                fprintf('iter = %3d, cost = %f\n', iter, cost_new);
                cost_old = cost_new;
            else 
                if mod(iter, 5) == 0
                    fprintf('.');
                end
                if mod(iter, 10) == 0 
                   fprintf('%d', iter);
                end     
            end        
        end 
        cost(iter) = calc_f(x_new);
%         figure,
%         plot(cost);
    end
    X = x_new;

end 
%% check gradient
% if ~check_grad(@calc_f, @grad, XX)
%     fprintf('Check gradient or cost again!\n')
%     pause
% end
%% Main Fista %%
opt.verbose = false;
opt.max_iter = 500;
opt.tol = 1e-8;

lambda = 0.1;
[XX,~] = fista(@grad, XX, L, lambda, opt);
cost = calc_F(XX);

% opts.lambda = 0.1;
% opts.max_iter = 300;
% opts.tol = 1e-8;
% opts.verbose = false;
% opts.L0 = 1;
% opts.eta = 2;
% 
% 
% XX = fista_backtracking(@calc_f, @grad, XX, opts, @calc_F);   
% cost = calc_F(XX);

[X,X0,xi,xij] = extractFromXX(XX);


end