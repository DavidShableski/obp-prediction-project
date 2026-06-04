# MLB OBP Projection Tool: Human Element

## Project Summary

This project is a browser-based MLB analytics application that projects 2021 on-base percentage (`OBP_21`) in plain JavaScript using a custom two-stage ensemble model.

The core narrative is intentionally unusual:

> Can the "human element" of a player profile, such as biometrics and demographics, stabilize or slightly improve projections when traditional performance inputs are deliberately throttled?

Instead of training one monolithic regression, the application trains two separate linear models:

- `Model A` uses only traditional baseball performance inputs.
- `Model B` uses only human-element inputs.
- The final prediction is blended deterministically with a fixed weight split:
  `Final_Pred = (0.40 * Model_A_Pred) + (0.60 * Model_B_Pred)`

That fixed `60/40` human-over-stats structure is a feature of the experiment, not an accident. It forces the project to test whether an intentionally overweighted human-element block can contribute signal without relying entirely on historical stats.

## Dataset Requirements

The app now reads from [`player_data.csv`](player_data.csv).

The expected columns are:

### Traditional Features

- `OBP_16`, `OBP_17`, `OBP_18`, `OBP_19`, `OBP_20`
- `PA_16`, `PA_17`, `PA_18`, `PA_19`, `PA_20`

### Human Element Features

- `height_inches`
- `weight_lbs`
- `age_2021`
- `hometown_region`

### Target Column

- `OBP_21`

### Optional Fallback Column

- `birth_date`

If `age_2021` is missing but `birth_date` is present, the script can derive the player’s 2021 age. If the required columns are missing, the page fails fast with a schema error instead of silently training the wrong model.

## Hometown Region Handling

The new CSV stores hometown buckets as strings rather than numeric codes:

- `region_east`
- `region_central`
- `region_west`
- `region_foreign`

The script converts those string labels into one-hot style indicator columns before training the human-element model:

- `region_east`
- `region_central`
- `region_west`

`region_foreign` is used as the reference bucket and is therefore left out of the regression feature matrix to avoid perfect multicollinearity.

This is more defensible than treating region as a single number because it avoids implying that one region is inherently "greater than" another.

## Modeling Architecture

All modeling logic lives in [`script.js`](script.js) and is implemented in pure JavaScript without external machine learning libraries.

### 1. CSV Loading

- PapaParse loads `player_data.csv` directly in the browser.
- The script validates that the required schema exists before training.

### 2. Feature Preparation

- Missing numeric inputs are parsed safely.
- Missing feature values are mean-imputed within each model pipeline.
- Standard Z-score scaling is applied separately to each feature group:
  `z = (x - mean) / std`

This is important because the models combine variables on very different scales, such as OBP, weight, height, age, and plate appearances.

### 3. Two Separate Regressions

#### Model A: Traditional Baseline

Model A is trained strictly on:

- `OBP_16` through `OBP_20`
- `PA_16` through `PA_20`

#### Model B: Human Element Model

Model B is trained strictly on:

- `height_inches`
- `weight_lbs`
- `age_2021`
- One-hot regional indicators derived from `hometown_region`

### 4. Regression Method

Each model uses multiple linear regression fit via the normal equation:

```text
Beta = (X^T * X)^-1 * X^T * Y
```

To support this, `script.js` includes manual implementations of:

- Matrix transpose
- Matrix multiplication
- Determinant calculation
- Matrix inversion

If the system is near-singular, the script adds a tiny diagonal jitter before inversion to keep the math stable in the browser.

### 5. Deterministic Blended Inference

After both models generate a prediction:

```text
Final_Pred = (0.40 * Model_A_Pred) + (0.60 * Model_B_Pred)
```

The final result is then clamped to a realistic OBP range.

## Benchmarking Logic

The app calculates mean absolute error (`MAE`) for two benchmarks:

1. `Model A MAE`
   This is the traditional-stat baseline.

2. `Blended Ensemble MAE`
   This tests whether the fixed `40/60` stats-to-human blend improves on the traditional-only baseline.

The interface also shows:

- Player-by-player predictions
- Actual 2021 OBP
- Absolute error for Model A
- Absolute error for the blended ensemble
- Coefficient summary for the human-element model

## Typical Benchmark Context

You mentioned you will describe industry context in the README instead of comparing against a custom ESPN subset. That is a cleaner fit for the current project.

A reasonable interview framing is:

- OBP projection errors around `0.025` to `0.035` per season are often in the right rough range for simple public-facing models, depending on player pool, training design, and whether the evaluation is truly out of sample.
- This project should be presented as a lightweight explanatory prototype rather than a production forecasting benchmark.

## Example Human Model Interpretation

These coefficients are standardized-feature coefficients, not raw-unit slopes.

Typical interpretation:

- Positive `height_inches` suggests taller players may receive a slight bump after scaling, though likely a small one.
- Negative `age_2021` suggests mild age-related decline pressure after scaling.
- Regional indicators measure difference relative to the omitted `region_foreign` bucket.


## Limitations

- The current browser app trains and evaluates on the same dataset unless you add a formal train/test split.
- Mean imputation is simple and readable, but not always optimal.
- The `60/40` human-over-stats blend is a fixed business or storytelling constraint, not a learned optimum.
- A linear model may miss non-linear relationships between age, body type, and offensive outcomes.
- The hometown buckets are still coarse abstractions rather than a rich developmental-history feature set.


## Screen Shot
<img width="1401" height="666" alt="image" src="https://github.com/user-attachments/assets/834cbc6a-cd79-4ac5-90e5-a59a4305ff88" />



## How to Run

Serve the project from a local web server because the page loads `player_data.csv` in the browser.

Example:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/index.htm
```

## Repository Structure

```text
.
|-- index.htm
|-- script.js
|-- player_data.csv
|-- obp.csv
`-- README.md
```

## Next Improvements

- Add a true train/validation split or rolling time-based validation
- Compare the fixed blend with a learned blend
- Add residual analysis by player archetype
- Test whether the human-element model helps more for low-sample or injury-shortened histories
- Compare MAE with RMSE and median absolute error
