
---

# **CHANGELOG**

## **v1.1.0**

* Added LOO-based automatic selection of polynomial order.
* Added LOO-based automatic q-norm truncation.
* Added LOO-based automatic selection of reduced dimensionality (PLS/PCA).

---

## **v1.0.0 — Initial Release**

This version provides the complete foundation of the **arbitrary Polynomial Chaos Expansion (aPCE)** framework, including polynomial construction, preprocessing, dimensionality reduction, regression modeling, prediction, uncertainty quantification, and sensitivity analysis.

### **aPCE Polynomial System**

* Supports two univariate polynomial generation approaches (2012 and 2018 versions)
* Moment-based orthogonal / orthonormal polynomial construction
* Multi-index generation with q-norm truncation
* Tensor-product multivariate polynomial basis

### **Data Preprocessing**

* Standardization using sample mean and standard deviation
* Automatic removal of zero-variance dimensions
* Consistent transformations for both training and prediction

### **Dimensionality Reduction**

* Supports PLS reduction
* Supports PCA reduction
* Reduced representation fully integrated with PCE basis construction and regression
* Shared projection used during both training and prediction

### **Regression Modeling**

Provides multiple regression methods for PCE coefficient estimation:

* OLS
* LARS
* Ridge
* Lasso

Suitable for both sparse and dense aPCE models.

### **Prediction**

* Applies the same standardization parameters used during training
* Automatically applies dimensionality reduction
* Reconstructs polynomial bases and produces model predictions

### **Uncertainty Quantification (UQ)**

Includes statistical outputs derived from the aPCE model:

* Mean
* Variance
* Standard deviation
* Skewness
* Kurtosis
* Confidence interval (CI)
* VaR / CVaR

### **Sobol Sensitivity Analysis**

Supports:

* First-order indices
* Second-order indices
* Total-effect indices
  With visualization support.

### **Visualization**

Includes:

* UQ distribution plots
* Sobol index plots
* PCE diagnostic plots

---