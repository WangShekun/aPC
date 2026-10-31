# aPC

MATLAB code for arbitrary polynomial chaos (aPC), scalar surrogate
modeling, uncertainty quantification, and Sobol sensitivity analysis.

Supports moment-based and Stieltjes polynomial bases, sparse regression,
dimension reduction, and model selection by leave-one-out (LOO) error.

## Requirements

MATLAB. Optional dependencies:

- UQLab: benchmark script `main.m`.
- Statistics and Machine Learning Toolbox: PLS, PCA, Ridge cross-validation, and Lasso.
- Parallel Computing Toolbox: parallel searches.
- Manopt: MCP dimension reduction.

## Quick start

From the repository root in MATLAB:

```matlab
addpath('core');
aPC;
rng(1);

data.X_train = 2 * rand(100, 2) - 1;
data.Y_train = data.X_train(:, 1).^2 + data.X_train(:, 2);

opts.Order = 2;
opts.qnorm = 1;
opts.aPC_Method = 'stieltjes';
opts.RegressionMethod = 'ols';
opts.Parallel = false;

[model, opts, data, reduction] = aPC_Train(data, opts);
Xtest = 2 * rand(20, 2) - 1;
Ypred = aPC_Predict(Xtest, model, opts, reduction);
```

Inputs use one sample per row: `X_train` is `N x D` and `Y_train` is `N x 1`.
The quick start does not require UQLab. For the UQLab benchmark, initialize the
paths as above, then run `main.m`.

## Options

`Order` sets the polynomial degree; `qnorm` controls hyperbolic truncation.
Use vectors to compare candidate values. Set `ReductionMethod` and `red_dim`
to enable dimension reduction, or `Parallel = true` for parallel searches.

`aPC_Method` accepts `'2012'`, `'2018'`, or `'stieltjes'`.
`RegressionMethod` accepts `'ols'`, `'lars'`, `'omp'`, `'ridge'`, or `'lasso'`.

Use `aPC_ComputeUQ(model, data)` for statistics and
`aPC_ComputeSensitivity(model, opts)` for Sobol indices.
