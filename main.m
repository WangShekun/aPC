clear; clc; close all;
rng(100);

aPC

ModelOpts = struct();
ModelOpts.Order    = [3];      % 最大多项式阶数
ModelOpts.qnorm    = [1];       % q-norm
ModelOpts.num_vars = 100;      % 输入维度

ModelOpts.ReductionMethod = 'mpldr';
ModelOpts.red_dim  = [40];

% ModelOpts.kernel_type   = {'gaussian'};
% ModelOpts.kernel_gamma  = [0.1];
% ModelOpts.kernel_degree = [2];
% ModelOpts.kernel_offset = [1];

% ModelOpts.label_kernel_type = {'gaussian'};
% ModelOpts.label_kernel_gamma  = [1/128^2];

ModelOpts.mpldr_opts.maxIter = 3000;

ModelOpts.mpldr_opts.whiten = 1;
ModelOpts.mpldr_opts.alpha_cov = 1.0;

ModelOpts.mpldr_opts.max_moment_order = [6];
ModelOpts.mpldr_opts.moment_base = 1.0;
ModelOpts.mpldr_opts.moment_decay = [0.4];

ModelOpts.mpldr_opts.lambda_y = [0.1];
ModelOpts.mpldr_opts.supervision_mode = 'pls';

ModelOpts.mpldr_opts.manopt_solver = 'rlbfgs';

ModelOpts.aPC_Method = '2012';      % 可选 '2012' | '2018' | 'stieltjes'
ModelOpts.RegressionMethod = 'lars';
ModelOpts.Parallel = 0;

uqlab

% InputOpts = ishigami_Input;

% InputOpts = wingweight_Input;

% InputOpts = morris10_Input;

InputOpts = dim100_Input;
% InputOpts = dim150_Input;
% InputOpts = dim200_Input;
% InputOpts = dim300_Input;

myInput = uq_createInput(InputOpts);

data = struct();
data.X_train = uq_getSample(myInput, 1000);
data.Y_train = dim100(data.X_train);

data.X_val   = uq_getSample(myInput, 10000);
data.Y_val   = dim100(data.X_val);

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

aPC_print(results, ModelOpts);

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
