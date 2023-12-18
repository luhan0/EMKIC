%Model
%    min sum_(v=1)^nV{alpha_v^r*[trace(H_v'D_vH_v)-lambda*tr(H_v'Y_v)]} +
%    gamma*||Y||_sp^p
%    s.t. H_v'H_v=I,v=1...nV, Y\in Ind, D=butterworth(S)

clear
% close all
addpath('.\datasets');
addpath('.\funs');

% try
%% ==================== Dataset ==========================
dataname = 'MSRC';
% dataname = 'Reuters';
% dataname = 'ORL'; %omega=0.01;
% dataname = 'mnist4'; omega=0.01;
% 
fprintf('----------- Dataset:【 %s 】---------------\n',dataname);
load([dataname '.mat']);
try
    gt=double(gt);
catch
    gt=double(Y);
end

nV = size(X, 2);
[dv, nN] = size(X{1}'); %[样本维数，总样本数目]
nC = length(unique(gt)); 
anchor_rate = 0.5;
%% ================ Parameter Setting ===========================
Iter_max = 200;
pho_mu = 1.2;
max_mu = 12e12;
resultFile = [ '.\results\model3_' dataname '.csv'];

All_lambda = [2:10];% [0.001 0.01 0.05 0.1 0.5];%
All_gamma = [1:10];
All_p = [0.1 0.3 0.5 0.7 0.9];
All_r = [3:10];
All_omega = 0.009;%[ 0.008 0.002 0.0001 0.0005 0.001];




% ================ Calculate D ================================
tic
N=2;
opt1. style = 1;
opt1. IterMax = 150;
opt1. toy = 0;
opt1. k = 10;

% 初始化
D = cell(1, nV); 
S = cell(1, nV);
Tri = cell(1, nV);
[~, C] = FastmultiCLR(X,nC,anchor_rate, opt1);
nM = floor(nN*anchor_rate);


time1 = toc;
for omega = All_omega
    for v = 1: nV
        Tri{v} = diag(sum(C{v}));
        S{v}=C{v}*pinv(Tri{v})*(C{v})';
        D{v} = sqrt(1./(1+(S{v}./omega).^(4)));
        D{v} = D{v}-diag(diag(D{v}));
        clear S 
    end

for p = All_p
for r = All_r
for lambda = All_lambda
for gamma = All_gamma
% ==================== main ==========================
    tic
    % ------- Init --------

    beta = 10;
    Y = cell(1,nV);H=Y;J=Y;Q=Y;
    for i = 1:nN
        H{1}(i,mod(i, nC)+1) = 1;
    end
    Y{1} = H{1};
    H{1} = H{1}./sqrt(sum(H{1}));
    for v = 1:nV
        Y{v} = zeros(nN,nC);
        H{v} = H{1};
        J{v} = zeros(nN,nC);
        Q{v} = zeros(nN,nC);
    end
    alpha = ones(1,nV)./nV;
    mu = 0.00001;
    
    % -------- iter --------
    obj = [1];
    iter = 1;
    while 1 
        % solve H
        for v = 1:nV
            objH = [1];
            iterH = 1;
            H_old = H{v};
            while 1
                Temp = (beta*eye(nN)-D{v})*H{v} + lambda*Y{v}./2;
                [U,~,V] = svd(Temp,"econ");
                H{v} = U*V';
                objH = [objH norm(H{v}-H_old,'fro')];
                if objH(iterH)<1e-8 || iterH >20
                    break
                end
                iterH = iterH + 1;
                H_old = H{v};
            end
            
        end
        clear Temp

        % solve Y
        for v = 1:nV
            Temp = J{v} - Q{v}./mu + (lambda*alpha(v)^r/mu)*H{v};
            Y{v} = ( Temp == max(Temp')' );
        end
        clear Temp

        % solve J Q mu
        QQ1=cell(1,nV);
        for v =1:nV
            QQ1{v} = Y{v} + Q{v}./mu;
        end
        Q_tensor = cat(3,QQ1{:,:});
        Qg = Q_tensor(:);
        sX=[nN, nC, nV];
        [myj, ~] = wshrinkObj_weight_lp(Qg,ones(1, nV)'.*(1*gamma/mu),sX, 0,3,p);
        J_tensor = reshape(myj, sX);
        for v=1:nV
            J{v} = J_tensor(:,:,v);
            Q{v} = Q{v} + mu*(Y{v}-J{v});
        end
        mu = min(pho_mu*mu, max_mu);

        %solve alpha
        for v = 1:nV
            Temp(v) = trace(H{v}'*D{v}*H{v}-lambda*H{v}'*Y{v});
%             Temp1(v) = trace(H{v}'*D{v}*H{v});
        end
        alpha = (Temp.^(1/(1-r)))./(sum(Temp.^(1/(1-r))));


        % Is converge
        obj_function = 0;
        for v = 1:nV
            obj_function = obj_function + norm(J{v}-Y{v},'fro');
        end
        obj = [obj obj_function];
        if abs(obj(iter))<1e-12 || iter > Iter_max
            break
        end
        iter = iter + 1;
        clear Temp

    end
    time = toc;
    YY = zeros(nN,nC);
    for v =1:nV
        YY = YY + alpha(v)^r*Y{v};
    end
    [~, label] = max(YY');
    result = ClusteringMeasure(gt,label);
    fprintf([' %.4f,%.4f,%.4f, lambda=%.3f,gamma=%.5f,omega=%.5f,' ...
        'r=%.1f,p=%.2f,iter=%d\n'],result(1:3),lambda,gamma,omega,r,p,iter)
    if ~exist(resultFile,'file')
       fid = fopen(resultFile,'w');
       fprintf(fid,'ACC,MIhat,Purity,P,R,F,RI,omega,lambda,gamma,r,p,beta,time,iter\n');
       fclose(fid);
    end
    fid = fopen(resultFile,'a');
    fprintf(fid,['%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,' ...
        '%.1f,%.1f,%d,%.2f,%d\n'],result,omega,lambda,gamma,r,p,beta,time,iter); 
    fclose(fid);

end
end
end
end
end