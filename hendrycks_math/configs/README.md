# Configurations
The following directory configuration files for replicating the experiments on the dataset MATH by Hendrycks et al. in Section 6.1 of the paper.
The directories:

* `hendrycks_math/configs/k_vs_pass_at_k/qwen-7b-baseline`
* `hendrycks_math/configs/k_vs_pass_at_k/qwen-7B-ifg`

Contain experiments for reprodcing the plots for K vs Pass-@K figures, and does not involve any training beyond hyperparameter tuning.

The directories 
* `hendrycks_math/configs/star/baseline_qwen_2.5_7b`
* `hendrycks_math/configs/star/ifg_qwen_2.5_7b`

Contain configuration for the STaR experiments. These experiments tune temperatures on a subset of the training dataset, run 5 iterations of STaR training, evaluating on the test dataset every iteration.
 
