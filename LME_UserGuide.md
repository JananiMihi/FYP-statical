# HRVBP-HyperNet: Linear Mixed Effects (LME) Model — R User Guide

**Project:** Exploratory Investigation of HR, BPD, and BPV as Predictors of Muscle Hypertrophy  
**University of Ruhuna | Department of Electrical and Information Engineering**  
**Script file:** `LME_HyperNet.R`

---

## Table of Contents

1. [What is a Linear Mixed Effects Model?](#1-what-is-a-linear-mixed-effects-model)
2. [Your Model Equation Explained](#2-your-model-equation-explained)
3. [Software Requirements](#3-software-requirements)
4. [Preparing Your Real Data](#4-preparing-your-real-data)
5. [Running the Script — Step by Step](#5-running-the-script--step-by-step)
6. [Understanding the Output](#6-understanding-the-output)
7. [Interpreting Key Results](#7-interpreting-key-results)
8. [Answering Your Research Questions](#8-answering-your-research-questions)
9. [Troubleshooting](#9-troubleshooting)
10. [Reporting Results in Your Paper](#10-reporting-results-in-your-paper)

---

## 1. What is a Linear Mixed Effects Model?

Your data has a **three-level nested structure**:

```
Level 3: Participant (N = 600)
   └── Level 2: Monthly measurement (9 time points)
          └── Level 1: Exercise/Muscle group (8 muscles)
```

Standard linear regression assumes all observations are **independent** — which yours are not. The same person is measured 9 times across 8 muscles, creating two types of correlation:

- **Within-participant correlation**: measurements from the same person are more similar to each other than to measurements from different people.
- **Temporal correlation**: measurements closer in time are more similar.

A **Linear Mixed Effects (LME) model** handles this by splitting predictors into two types:

| Type | What it captures | Example |
|------|-----------------|---------|
| **Fixed effects** (β, γ, δ) | Population-level relationships that apply to everyone | "PC_HR1 increases hypertrophy by β% on average" |
| **Random effects** (u₀, u₁) | Individual deviations from the population average | "Participant 42 grows 3% faster than average" |

---

## 2. Your Model Equation Explained

From Section 3.0.9 of your report:

```
H(i,e)(t) = z(i)(t)ᵀ β  +  x(i)ᵀ γ  +  Σ δ_jl × w(i)_jl(t)  +  u(i)_0  +  u(i)_1 × t  +  ε(i,e)(t)
```

| Symbol | Meaning | In R |
|--------|---------|------|
| `H(i,e)(t)` | Hypertrophy % for participant i, exercise e, at month t | `H` column |
| `z(i)(t)ᵀ β` | Cardiovascular PCA components × their coefficients | `PC_HR1 + PC_HR2 + PC_HR3 + PC_BP1 + PC_BP2` |
| `x(i)ᵀ γ` | Demographic/nutritional covariates | `sex + age + bmi + protein_intake + sleep_psqi + stress_pss + group` |
| `Σ δ_jl × w(i)_jl` | Interaction terms (CV component × binary covariate) | `w_PC_HR1_x_sex` etc. |
| `u(i)_0` | Random intercept per participant (individual baseline muscle size) | `(1 \| participant_id)` |
| `u(i)_1 × t` | Random slope per participant (individual growth rate) | `(0 + time \| participant_id)` |
| `ε(i,e)(t)` | Residual error ~ N(0, σ²) | Residuals |

In **lmer syntax** (the R function you use):

```r
H ~ PC_HR1 + PC_HR2 + PC_HR3 + PC_BP1 + PC_BP2 +    # cardiovascular z(i)(t)
    sex + age + bmi + protein_intake +                 # demographics x(i)
    sleep_psqi + stress_pss + factor(group) + time +   # more covariates
    w_PC_HR1_x_sex + w_PC_HR1_x_nutrition_cat + ...   # interactions w(i)_jl
    (1 + time | participant_id) +                      # random intercept + slope
    (1 | exercise)                                     # random exercise effect
```

---

## 3. Software Requirements

### 3.1 Install R and RStudio

| Software | Download |
|----------|----------|
| R (≥ 4.2.0) | https://cran.r-project.org |
| RStudio Desktop (free) | https://posit.co/downloads |

### 3.2 Required Packages

The script automatically installs all packages. Core packages are:

| Package | Purpose |
|---------|---------|
| `lme4` | Core LME model fitting (`lmer()` function) |
| `lmerTest` | Adds p-values to `lme4` output (Satterthwaite df) |
| `dplyr`, `tidyr` | Data manipulation |
| `ggplot2` | All visualisations |
| `car` | VIF multicollinearity check |
| `glmnet` | Elastic net feature selection |
| `MuMIn` | R-squared for mixed models |
| `broom.mixed` | Tidy model output tables |

---

## 4. Preparing Your Real Data

### 4.1 Required Dataset Structure

Your dataset must be in **long format** — one row per participant × month × exercise:

| Column Name | Type | Description | Example |
|-------------|------|-------------|---------|
| `participant_id` | integer | Unique participant ID | 1 – 600 |
| `time` | integer | Month number | 1 – 9 |
| `exercise` | character | Exercise name | "bicep_curl" |
| `H` | numeric | Hypertrophy index % (from Eq. 3.31) | 5.2 |
| `group` | integer | Activity group (1–4) | 1 |
| `sex` | integer | 0=female, 1=male | 1 |
| `age` | numeric | Age in years | 24 |
| `bmi` | numeric | BMI kg/m² | 23.5 |
| `protein_intake` | numeric | g/kg/day | 1.8 |
| `sleep_psqi` | integer | PSQI score (0–21) | 6 |
| `stress_pss` | integer | PSS-10 score (0–40) | 14 |
| `nutrition_cat` | integer | 0 = below, 1 = ≥1.6g/kg protein | 1 |
| `PC_HR1` | numeric | 1st HR PCA component | 0.84 |
| `PC_HR2` | numeric | 2nd HR PCA component | -0.32 |
| `PC_HR3` | numeric | 3rd HR PCA component | 0.11 |
| `PC_BP1` | numeric | 1st BP PCA component | 0.56 |
| `PC_BP2` | numeric | 2nd BP PCA component | -0.19 |

> **Note:** `PC_HR*` and `PC_BP*` are the outputs from your PCA step (Section 3.0.8 of report). Run PCA on the raw HRV/BPV metrics first, then export the component scores as columns.

### 4.2 Replacing the Simulated Data

Find this block near the top of `LME_HyperNet.R` (around line 50):

```r
# SIMULATE DATASET (Replace with your actual data import)
set.seed(42)
...
```

Replace the entire simulation block with your actual data import:

```r
# Option A: From CSV file
data <- read.csv("C:/Your/Path/hypertrophy_longitudinal_data.csv")

# Option B: From Excel file
# install.packages("readxl")
library(readxl)
data <- read_excel("C:/Your/Path/hypertrophy_data.xlsx", sheet = "LongFormat")

# Option C: From SPSS file
# install.packages("haven")
library(haven)
data <- read_sav("C:/Your/Path/hypertrophy_data.sav")

# Rename 'data' to 'long_data' to match the rest of the script
long_data <- data
```

### 4.3 Converting Wide to Long Format

If your data is in **wide format** (one row per participant, one column per month), convert it:

```r
library(tidyr)

# Example: columns are H_month1, H_month2, ..., H_month9
long_data <- wide_data %>%
  pivot_longer(
    cols      = starts_with("H_month"),
    names_to  = "time",
    values_to = "H"
  ) %>%
  mutate(time = as.integer(gsub("H_month", "", time)))
```

---

## 5. Running the Script — Step by Step

### Step 1: Open the Script

1. Open **RStudio**
2. Go to **File → Open File**
3. Select `LME_HyperNet.R`

### Step 2: Set Working Directory

In RStudio Console, type:

```r
setwd("C:/path/to/your/project/folder")
```

Or use **Session → Set Working Directory → To Source File Location** in the menu.

### Step 3: Run the Script

Either:
- Click **Source** button (top right of the script window) to run everything at once, OR
- Press **Ctrl+Enter** (Windows/Linux) or **Cmd+Return** (Mac) to run line by line

### Step 4: Check Outputs

After running, your folder will contain:

| File | Contents |
|------|----------|
| `plot_01_spaghetti.png` | Individual hypertrophy trajectories by group |
| `plot_02_resid_fitted.png` | Residuals vs fitted (homoscedasticity check) |
| `plot_03_qq.png` | Q-Q plot (normality of residuals) |
| `plot_04_random_effects.png` | BLUPs: individual intercepts vs slopes |
| `plot_05_forest.png` | Forest plot of all fixed effects |
| `plot_06_pred_obs.png` | Predicted vs Observed on test set |
| `results_fixed_effects.csv` | All β coefficients with SE, t, p-values |
| `results_spearman_correlations.csv` | ρs for each CV component |
| `results_per_muscle_rmse.csv` | RMSE breakdown per muscle group |
| `results_model_comparison.csv` | AIC/BIC for all 5 models |
| `results_random_effects_blup.csv` | Individual random effect estimates |

---

## 6. Understanding the Output

### 6.1 Fixed Effects Table

```
Fixed effects:
               Estimate Std. Error      df  t value Pr(>|t|)
(Intercept)    1.2450    0.3210   482.1    3.879   0.0001 ***
PC_HR1         0.8920    0.0442  3820.5   20.181   <2e-16 ***
PC_BP1         0.3210    0.0381  3815.2    8.425   <2e-16 ***
sex            0.4510    0.2890   597.3    1.561   0.1189
time           0.4820    0.0112  3800.1   43.036   <2e-16 ***
```

**How to read this:**
- **Estimate**: The β coefficient. `PC_HR1 = 0.892` means: for every 1-unit increase in the first HR PCA component, hypertrophy increases by **0.892%**, holding all other variables constant.
- **Std. Error**: Precision of the estimate. Smaller = more precise.
- **t value**: Estimate ÷ Std. Error. |t| > 2 is a rough guide for significance.
- **Pr(>|t|)**: p-value. Stars: `***` p<0.001, `**` p<0.01, `*` p<0.05

### 6.2 Random Effects Structure

```
Random effects:
 Groups         Name        Variance Std.Dev. Corr
 participant_id (Intercept)  4.210   2.052
                time          0.018   0.134   -0.21
 exercise       (Intercept)  1.340   1.158
 Residual                    2.890   1.700
```

**How to read this:**
- **participant_id Variance (Intercept) = 4.21**: Participants differ considerably in their baseline muscle size (baseline hypertrophy).
- **participant_id Variance (time) = 0.018**: Participants also differ in their **growth rate** over time (the random slope).
- **Corr = -0.21**: Participants who start with higher baseline hypertrophy tend to grow slightly slower (negative correlation between u₀ and u₁).
- **Residual Variance = 2.89**: Unexplained within-participant variation.

### 6.3 R-Squared Values

```
Marginal  R²:    0.312  (31.2%)   ← variance explained by fixed effects alone
Conditional R²:  0.718  (71.8%)   ← variance explained by fixed + random effects
```

- **Marginal R²**: How much of the variance in hypertrophy is explained by your CV predictors + demographics alone.
- **Conditional R²**: Total variance explained when individual differences (random effects) are included.
- The gap (71.8% − 31.2% = 40.6%) is the variance attributable to individual participant differences.

### 6.4 Likelihood Ratio Test Output

```
        Df    AIC     BIC   logLik deviance  Chisq Chi Df Pr(>Chisq)
model_2  8  42100   42180   -21042    42084
model_3 10  41850   41950   -20915    41830  254.3      2  < 2.2e-16 ***
```

**How to read this:**
- **Chisq = 254.3**: The improvement in model fit when adding BP components.
- **Pr(>Chisq) < 2.2e-16**: Adding BP components **significantly improves** prediction (p < 0.05). This answers **RQ2**.

---

## 7. Interpreting Key Results

### 7.1 Cardiovascular Predictors (RQ1)

**Spearman correlation output:**
```
  Variable    rho      p_value  Significant_0.3
  PC_HR1    0.342   < 2e-16    TRUE             ← qualifies (|ρ| ≥ 0.3)
  PC_HR2   -0.118    0.0023    FALSE
  PC_BP1    0.298    < 2e-16   FALSE            ← borderline; check exact p
```

Report as: *"PC_HR1 exhibited a statistically significant Spearman correlation with hypertrophy (ρs = 0.342, p < 0.001), meeting the pre-specified threshold of |ρs| ≥ 0.3."*

### 7.2 Incremental Value of BP (RQ2)

Look at the LRT output for **Model 2 vs Model 3**:
- If p < 0.05 → Adding BPD/BPV significantly improves prediction → **RQ2 = YES**
- If p ≥ 0.05 → BP components do not add significant incremental value

### 7.3 Group Moderation (RQ5)

Look at the LRT output for **Model 3 vs Model 4** (interaction terms):
- If p < 0.05 → Interaction between CV components and group is significant → **RQ5 = YES**
- Check individual interaction coefficient p-values in the fixed effects table.

---

## 8. Answering Your Research Questions

| RQ | Where to look in output | Criterion |
|----|------------------------|-----------|
| **RQ1** — CV predictors of hypertrophy | Spearman table + Fixed effects β | \|ρs\| ≥ 0.3, p < 0.05 |
| **RQ2** — Incremental value of BP | LRT: Model 2 vs Model 3 | p < 0.05 → BP adds value |
| **RQ4** — LME vs ensemble superiority | Compare RMSE: LME vs MERF/GPBoost/ODE-LSTM | Lower RMSE = better |
| **RQ5** — Group moderation | LRT: Model 3 vs Model 4; δ_jl p-values | Interaction p < 0.05 |

---

## 9. Troubleshooting

### "Model failed to converge"

```r
# Try different optimiser
control = lmerControl(optimizer = "Nelder_Mead", optCtrl = list(maxfun = 2e6))
# OR
control = lmerControl(optimizer = "nlminbwrap")
```

### "Singular fit warning"

This means one of your random effects explains almost zero variance. Try simplifying:
```r
# Remove random slope, keep only random intercept
H ~ [fixed] + (1 | participant_id) + (1 | exercise)
```

### "NAs in predictors"

```r
# Check missing values
colSums(is.na(long_data))

# Remove rows with NAs (or use imputation)
long_data <- na.omit(long_data)
```

### Package installation fails

If you are on a restricted university network:
```r
# Try installing with different CRAN mirror
install.packages("lme4", repos = "https://cloud.r-project.org")
```

---

## 10. Reporting Results in Your Paper

### Standard LME Reporting Template

> *"A linear mixed-effects model was fitted using the `lme4` package (v1.1-x) in R (v4.x), with participant-level random intercepts and random slopes for time to account for individual baseline differences and growth trajectories. Fixed effects comprised the PCA-reduced cardiovascular components (PC_HR1–3, PC_BP1–2), demographic covariates (sex, age, BMI, protein intake, PSQI, PSS-10, activity group), time, and interaction terms between qualifying CV components (|ρs| ≥ 0.3) and binary covariates (sex, nutrition category). Denominator degrees of freedom were computed using the Satterthwaite approximation (lmerTest package). Marginal and conditional R² were computed using the method of Nakagawa & Schielzeth (MuMIn package). Model comparison was performed via likelihood ratio tests (REML = FALSE)."*

### Reporting a Fixed Effect

> *"Each 1-unit increase in the first principal HR component (PC_HR1) was associated with a β = 0.892% (95% CI: 0.806, 0.978; t(3820) = 20.18, p < 0.001) increase in muscle hypertrophy, after controlling for all demographic, nutritional, and training covariates."*

### Reporting Model Fit

> *"The primary model achieved marginal R² = 0.312 and conditional R² = 0.718, indicating that cardiovascular and demographic fixed effects explained 31.2% of hypertrophy variance, while the inclusion of individual random effects increased total explained variance to 71.8%. On the held-out test set (N = 120 participants), the model yielded RMSE = X.XX%, MAE = X.XX%, and Pearson r = X.XX."*

---

*For questions, refer to the methodology in Section 3.0.9 of the project proposal report.*
