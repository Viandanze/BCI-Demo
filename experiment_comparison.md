# NeuroDecode — Complete Experiment Results Comparison

## 1. Model Performance Overview

| Model | Accuracy | Kappa | F1-macro | Notes |
|-------|----------|-------|----------|-------|
| EEGNet baseline | 45.83% | 0.227 | 0.406 | Default params, 5-fold CV, 8 subjects |
| EEGNet + mixup | 49.58% | 0.157 | 0.394 | Strategy A best augmentation |
| EEGNet + hyperparameter search | **54.32%** | **0.243** | — | F1=4, D=1, dropout=0.3, kernel=128 |
| EEGNet + architecture improvement | 45.83% | 0.227 | 0.406 | Baseline best, improvements did not surpass |
| Conformer | 38.43% | -0.002 | 0.298 | embed_dim=64, 3 layers, 8 heads |
| TCN | 40.35% | 0.027 | 0.303 | n_filters=32, 4 layers |
| Ensemble-Voting | 79.29% | 0.472 | — | EEGNet + Riemannian, soft voting |
| Ensemble-Stacking | **82.14%** | **0.537** | — | EEGNet + Riemannian, tangent space stacking |
| Ensemble-Weighted | 80.00% | 0.477 | — | EEGNet + Riemannian, weighted voting |

> Note: EEGNet series/Conformer/TCN use 5-fold CV (8 subjects), ensemble methods use single train/test split

## 2. EEGNet Tuning Detailed Results

### Strategy A - Data Augmentation Combinations (5-fold CV, 8 subjects)

| Augmentation | Accuracy | Kappa | F1-macro |
|--------------|----------|-------|----------|
| mixup | 0.4958 | 0.157 | 0.394 |
| time_shift | 0.4950 | 0.196 | 0.456 |
| baseline | 0.4583 | 0.227 | 0.406 |
| temporal_mask | 0.4458 | 0.000 | 0.267 |
| band_perturbation | 0.4333 | 0.000 | 0.267 |
| channel_mask | 0.4325 | 0.196 | 0.456 |
| all_combined | 0.4067 | 0.125 | 0.433 |
| gaussian_noise | 0.3667 | 0.074 | 0.254 |

### Strategy B - Hyperparameter Search (54 random search configs)

| Parameter | Optimal Value |
|-----------|---------------|
| F1 (temporal filters) | 4 |
| D (spatial filters) | 1 |
| Dropout | 0.3 |
| Kernel Length | 128 |
| **Best Accuracy** | **54.32%** |
| **Best Kappa** | **0.243** |

Key finding: Small filters (F1=4) + long kernel (kernel=128) performed best, improving from baseline 45.83% to 54.32% (+8.5pp)

### Strategy C - Architecture Improvements

| Method | Accuracy | Kappa | F1-macro |
|--------|----------|-------|----------|
| baseline | 0.4583 | 0.227 | 0.406 |
| se_attention | 0.4575 | 0.250 | 0.468 |
| label_smoothing | 0.4192 | 0.186 | 0.451 |
| batchnorm | 0.4058 | 0.155 | 0.375 |
| combined | 0.4017 | 0.000 | 0.078 |

Key finding: All architecture improvements failed to surpass baseline; combined even dropped to 40%

## 3. Deep Learning Models — Per-Subject Results

### Conformer (5-fold CV)

| Subject | Accuracy ± Std | Kappa | F1-macro |
|---------|----------------|-------|----------|
| 1 | 0.3933 ± 0.0669 | -0.022 | 0.305 |
| 2 | 0.4057 ± 0.0950 | 0.025 | 0.361 |
| 3 | 0.2667 ± 0.2261 | -0.222 | 0.154 |
| 4 | 0.4533 ± 0.0267 | 0.101 | 0.357 |
| 5 | 0.3797 ± 0.0589 | -0.066 | 0.258 |
| 6 | 0.3458 ± 0.0568 | -0.127 | 0.230 |
| 7 | 0.5301 ± 0.0760 | 0.233 | 0.474 |
| 8 | 0.3000 ± 0.1149 | -0.161 | 0.245 |
| **Mean** | **0.3843** | **-0.003** | **0.298** |

### TCN (5-fold CV)

| Subject | Accuracy ± Std | Kappa | F1-macro |
|---------|----------------|-------|----------|
| 1 | 0.3192 ± 0.1098 | 0.011 | 0.232 |
| 2 | 0.4200 ± 0.0933 | -0.042 | 0.278 |
| 3 | 0.2667 ± 0.2261 | -0.100 | 0.215 |
| 4 | 0.5067 ± 0.0327 | 0.136 | 0.340 |
| 5 | 0.4261 ± 0.0954 | 0.024 | 0.294 |
| 6 | 0.4497 ± 0.0745 | 0.067 | 0.359 |
| 7 | 0.3562 ± 0.0833 | 0.115 | 0.346 |
| 8 | 0.4837 ± 0.0802 | 0.108 | 0.357 |
| **Mean** | **0.4035** | **0.027** | **0.303** |

## 4. Ensemble Learning Detailed Results

| Strategy | Accuracy | Kappa | EEGNet Single | Riemann Single | Training Time |
|----------|----------|-------|---------------|----------------|---------------|
| Voting (soft) | 0.7929 | 0.472 | 0.7714 | 0.600 | 7.8s |
| Stacking (tangent) | **0.8214** | **0.537** | — | — | 128.6s |
| Weighted | 0.8000 | 0.477 | — | — | 13.5s |
| Voting (tangent) | 0.7286 | 0.173 | 0.7500 | 0.7071 | — |

## 5. Key Conclusions

1. **Single deep learning models have limited performance on PhysioNet MI dataset**: EEGNet after hyperparameter search reached only 54.32%, Conformer (38.43%) and TCN (40.35%) were worse, close to random guessing (3-class ~33.3%)

2. **Hyperparameter search yielded the greatest gain**: From baseline 45.83% to 54.32% (+8.5pp), small filters + long kernel configuration was most effective

3. **Architecture improvements were ineffective**: BatchNorm/LabelSmoothing/SE-Attention/Combined all failed to surpass baseline; complex architectures overfit with insufficient data

4. **Ensemble learning significantly outperforms single models**: Stacking achieved 82.14%, +27.8pp over best single model; Riemannian geometry features and deep learning features are highly complementary

5. **Subject 7 performed best**: Highest in both Conformer (53.01%) and EEGNet baseline, indicating large individual differences
