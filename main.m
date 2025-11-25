clear; clc; close all;
rng(70);

addpath(genpath('basis'));
addpath(genpath('coefficients'));
addpath(genpath('display'));
addpath(genpath('utils'));
addpath(genpath('predict'));
addpath(genpath('UQ'));
addpath(genpath('sensitivity'));
addpath(genpath('dimreduct'));
addpath(genpath('benchmark'));
addpath(genpath('apc_train'));

ModelOpts = struct();
ModelOpts.Order    = 3:6;      % 最大多项式阶数
ModelOpts.qnorm    = 0.7;       % q-norm
ModelOpts.num_vars = 20;       % 输入维度
ModelOpts.ReductionMethod = 'pls';
ModelOpts.red_dim  = 5:15;
ModelOpts.aPC_Method = '2012';      % 可选 '2012' | '2018'
ModelOpts.RegressionMethod = 'LARS';

uqlab

% InputOpts.Marginals(1).Type       = 'Gaussian';
% InputOpts.Marginals(1).Parameters = [3 0.2];
% InputOpts.Marginals(2).Type       = 'Uniform';
% InputOpts.Marginals(2).Parameters = [0 2];
% InputOpts.Marginals(3).Type       = 'Gaussian';
% InputOpts.Marginals(3).Parameters = [3 0.1];

% InputOpts.Marginals(1).Type = 'Uniform';
% InputOpts.Marginals(1).Parameters = [150 200];
% InputOpts.Marginals(2).Type = 'Uniform';
% InputOpts.Marginals(2).Parameters = [220 300];
% InputOpts.Marginals(3).Type = 'Uniform';
% InputOpts.Marginals(3).Parameters = [6 10];
% InputOpts.Marginals(4).Type = 'Uniform';
% InputOpts.Marginals(4).Parameters = [-10 10];
% InputOpts.Marginals(5).Type = 'Uniform';
% InputOpts.Marginals(5).Parameters = [16 45];
% InputOpts.Marginals(6).Type = 'Uniform';
% InputOpts.Marginals(6).Parameters = [0.5 1];
% InputOpts.Marginals(7).Type = 'Uniform';
% InputOpts.Marginals(7).Parameters = [0.08 0.18];
% InputOpts.Marginals(8).Type = 'Uniform';
% InputOpts.Marginals(8).Parameters = [2.5 6];
% InputOpts.Marginals(9).Type = 'Uniform';
% InputOpts.Marginals(9).Parameters = [1700 2500];
% InputOpts.Marginals(10).Type = 'Uniform';
% InputOpts.Marginals(10).Parameters = [0.025 0.08];

for k = 1:20
    InputOpts.Marginals(k).Type = 'Uniform';
    InputOpts.Marginals(k).Parameters = [0 1];
end

% for ii = 1:100
%     InputOpts.Marginals(ii).Type = 'Uniform';
%     if ii == 20
%         InputOpts.Marginals(ii).Parameters = [1 3];
%     else
%         InputOpts.Marginals(ii).Parameters = [1 2];
%     end
% end

myInput = uq_createInput(InputOpts);

data = struct();
data.X_train = uq_getSample(myInput, 1000);
data.Y_train = morris10(data.X_train);

data.X_val   = uq_getSample(myInput, 2000);
data.Y_val   = morris10(data.X_val);

%% 
MetaOpts_OLS.Type = 'MetaModel';
MetaOpts_OLS.MetaType = 'PCE';
MetaOpts_OLS.Method = 'LARS';
MetaOpts_OLS.TruncOptions.qNorm = 1;
MetaOpts_OLS.Degree = 3;
MetaOpts_OLS.ExpDesign.X = data.X_train;
MetaOpts_OLS.ExpDesign.Y = data.Y_train;
MetaOpts_OLS.Display = 0;
myPCE_OLS = uq_createModel(MetaOpts_OLS);

Y_OLS = uq_evalModel(myPCE_OLS, data.X_val);
RELMSE_OLS = sum((Y_OLS - data.Y_val).^2) / sum((mean(data.Y_val) - data.Y_val).^2)

% PCESobol.Type = 'Sensitivity';
% PCESobol.Method = 'Sobol';
% PCESobol.Sobol.Order = 2;
% PCESobolAnalysis = uq_createAnalysis(PCESobol);
% % uq_display(PCESobolAnalysis);
% uq_print(PCESobolAnalysis);
%% 
[results, ModelOpts, data, ReduceModel] = aPC_Train(data, ModelOpts);

data.Y_pred = aPC_Predict(data.X_val, results, ModelOpts, ReduceModel);

results.RelMSE = sum((data.Y_pred - data.Y_val).^2) / ...
                 sum((data.Y_val - mean(data.Y_val)).^2);

aPC_print(results, ModelOpts, ModelOpts.Indices);

% aPC_plot(data, results, ModelOpts, "all");

%% 不确定性量化
UQ = aPC_ComputeUQ(results, data);
aPC_printUQ(UQ);
% aPC_PlotUQ(UQ, data);

aPC_error = [abs(UQ.mean-mean(data.Y_val))/mean(data.Y_val) ...
    ;abs(UQ.var-var(data.Y_val))/var(data.Y_val)]
PCE_error = [abs(myPCE_OLS.PCE.Moments.Mean-mean(data.Y_val))/mean(data.Y_val) ...
    ;abs(myPCE_OLS.PCE.Moments.Var-var(data.Y_val))/var(data.Y_val)]

%% 灵敏度分析
% SA = aPC_ComputeSensitivity(results, ModelOpts);
% % aPC_PlotSensitivity(SA, ModelOpts);
% aPC_printSensitivity(SA,ModelOpts);
