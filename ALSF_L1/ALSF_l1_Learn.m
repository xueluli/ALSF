function [D_c,D0,Omega_c,Omega0] = ALSF_l1_Learn(Y_train, label_train, K_train, K0)

StartT = tic;
ki = []; %% No. of collums corresponding each class in the dictionary
C = max(label_train); %% No. of classes
% for i = 1:C
%     ki = [ki sum(label_train == i)];
% end
for i = 1:C
    ki = [ki K_train];
end
[row, col] = size(Y_train); %% the No. of the row and the collums of D
N_train = length(label_train)/C;

Ki =  ki(1); %% assuming that the No. of collums are the same for each class
Y_range = label_to_range(label_train);
D_range = Ki*(0:C);
% K0 = 3; %% No. of collums corresponding to D0

%% == Parameters needed for computing initial values == %%
% Using the same method in the referenced paper to obtain the initial values
% refereced paper “”Fast Low-rank Shared Dictionary Learning for Image
% Classification

opts.D_range = D_range;
opts.initmode = 'normal';
opts.k0 = K0;
opts.lambda1 = 0.0001;
opts.lambda2 = 0.01;
opts.lambda3 = 0.00;
opts.max_iter = 30;
opts.show_progress = true;
opts.rank_flag = 1;
opts = initOpts(opts);
opts.tol = 1e-10;
optsinit = opts;
optsinit.max_iter = 30;
optsinit.verbose = 0;

[D_ini, D0_ini, X_ini, X0_ini] = LRSDL_init(Y_train, Y_range, D_range, opts);

D_total = [D_ini D0_ini]; % initial D_total = [D D0]
Omega_total = pinv(D_total); % initial Omega_total = pinv(D_total);
Omega_ini = Omega_total(1:Ki*C,:); % initial Omega
Omega0_ini = Omega_total(Ki*C+1:Ki*C+K0,:); % initial Omega0


%% ================= Parameter setting for iterations ================= %%
tau = 0.1;
% lambda = 0.1;
lambda1 = 0.1;
lambda2 = 0.1;
lambda3 = 0.1;
lambda4 = 0.1;
eita1 = 0.1;
eita2 = 0.1;
Max_iter = 18;
rho = 0.1;
%% packing the data %%
for c = 1:C
    p1 = (c-1)*N_train;
    p2 = (c-1)*Ki;
%     X{c} = X_ini(:,p+1:p+N_train);
    Y{c} = Y_train(:,p1+1:p1+N_train);
%     X0{c} = X0_ini(:,p+1:p+N_train);
    Omega{c} = Omega_ini(p2+1:p2+Ki,:);
    D{c} = D_ini(:,p2+1:p2+Ki);
end
XX = [X_ini;X0_ini];
Ym = mean(Y_train,2); %% The average value of the observed data
D0 = D0_ini;
Omega0 = Omega0_ini;
% 
% L = 2450000; % TCGA 200
% L = 4000000; % Breast 170 
% L = 8500000; % Breast 340 
% L = 7500000; % Kidney 320
%  L = 5500000; % Kidney 240
% L = 2500000; % Kidney 240
%  L = 1500000; % Kidney 80
% L = 3000000; % Kidney 160
% L = 13000000; % Breast
L = 1500000; %TCGA %Spleen
% L = max(eig(D_ini'*D_ini+2*lambda2*eye(size(D_ini,2))))+max(eig(D0'*D0+lambda2*eye(size(D0,2))))+3;
% Lipschitz constant of the gradient (test)


optX.lambda2 = lambda2;
optX.lambda3 = lambda3;
optX.lambda4 = lambda4;
optX.tau = tau;
optX.K0 = K0;
endT = toc(StartT);
%% ================= Updating process ================= %%
XX_old = XX;
for tt = 1:Max_iter
    tt

    %% == updating X == %%
    %%% calculate the gradient of X %%%
    [XX,~,X0,Xi,~] = XX0_update(XX,D,D0,Y,Omega,Omega0,C,N_train,L,Ki,optX);
    e = norm1(XX-XX_old)/numel(XX);
    if e < 1e-8
        break;
    end
    
    for i = 1:C
        %% == X_i == %%
        %         Z1 = [Y{i}-D0*X0{i}; sqrt(tau*lambda1)*Omega{i}*Y{i}];
        %         P1 = [D{i};sqrt(tau*lambda1)*eye(Ki)];
        %         X{i} = pinv(P1)*Z1;
        
        %% == X_0i == %%
        %         Z2 = [Y{i}-D{i}*X{i}; sqrt(tau*lambda2)*Omega0*Y{i}];
        %         P2 = [D0; sqrt(tau*lambda2)*eye(K0)];
        %         X0{i} = pinv(P2)*Z2;
        
        %% == Omega_i == %%
        P3 = [];
        if i == 1
            P3 = Y_train(:,N_train+1:end);
            Xij = XX(1:Ki,N_train+1:end);
        else
            P3 = [Y_train(:,1:(i-1)*N_train) Y_train(:,i*N_train+1:end)];
            Xij = [XX((i-1)*Ki+1:i*Ki,1:(i-1)*N_train) XX((i-1)*Ki+1:i*Ki,i*N_train+1:end)];
        end
        P3 = [1/sqrt((C-1))*P3 sqrt(lambda4)*P3 sqrt(lambda2)*Y{i} sqrt(eita1)*eye(row)];
        Z3 = [zeros(Ki, (C-1)*N_train) sqrt(lambda4)*Xij sqrt(lambda2)*Xi{i} zeros(Ki,row)];
        Omega{i} = Z3*pinv(P3);
        
        %% == D_i == %%
        D{i} = (Y{i}-D0*X0{i})*pinv(Xi{i});
%         size(D{i})
    end
    %% == Omega_0 == %%
    P4 = [];
    Z4 = [];
    for c = 1:C
        P4 = [P4 Y{c} sqrt(lambda3/lambda2)*(Y{c}-Ym)];
        Z4 = [Z4 X0{i} zeros(K0,N_train)];
    end
    P4 = [P4 sqrt(eita2)*eye(row)];
    Z4 = [Z4 zeros(K0,row)];
    Omega0 = Z4*pinv(P4);
    %% == D_0 == %%
    P5 = [];
    Z5 = [];
    
    for c = 1:C       
        P5 = [P5 X0{c}];
        Z5 = [Z5 Y{c}-D{c}*Xi{c}];
    end
    
    threshold = eita2/rho*0.01;
    D0 = update_D0(Z5*pinv(P5),threshold);
    XX_old = XX;
end

Omega_c = [];
D_c = [];
for c = 1:C
    Omega_c = [Omega_c; Omega{c}];
    D_c = [D_c D{c}];
end
% pred_label = JSLF_pred(D_c,D0,Omega_c,Omega0,Y_test,label_train);
% acc = calc_acc(pred_label, label_test);
end
