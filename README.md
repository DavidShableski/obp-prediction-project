# MLB On-Base Percentage Projection Tool

## Project Objective

This project is a small browser-based sports analytics tool that predicts MLB players' 2021 on-base percentage (OBP) using player data from the 2016 through 2020 seasons. It loads a local CSV file in the browser, trains a linear regression model in JavaScript, compares each prediction with the actual 2021 OBP, and displays the prediction error.

The goal is to keep the project simple and readable while expanding the original weighted-average projection into a basic statistical model.

## Data Used

The project uses [`obp.csv`](obp.csv), which includes player-level OBP and plate appearance data. Important columns include:

- Player information: `Name`, `playerid`, `birth_date`
- Historical OBP features: `OBP_16`, `OBP_17`, `OBP_18`, `OBP_19`, `OBP_20`
- Historical plate appearance features: `PA_16`, `PA_17`, `PA_18`, `PA_19`, `PA_20`
- Target value: `OBP_21`

The script also calculates each player's age for 2021 from `birth_date`.

## Methodology

The main logic is in [`script.js`](script.js):

1. The browser fetches `obp.csv`.
2. PapaParse reads the CSV and converts it into JavaScript objects.
3. The script builds training rows from players with a valid `OBP_21`, a valid birth date, and at least two seasons of usable historical OBP and plate appearance data.
4. Missing historical feature values are filled with the average value from the training data.
5. Features are standardized so OBP, plate appearances, and age can be used together more safely.
6. A multiple linear regression model is trained directly in JavaScript using the normal equation.
7. The trained model predicts each player's 2021 OBP.
8. Predictions are compared with actual `OBP_21` values.

No external machine learning framework is used. The small amount of matrix math needed for regression is implemented directly in plain JavaScript.

## Features Used by the Model

The regression model uses these inputs:

- `OBP_16`
- `OBP_17`
- `OBP_18`
- `OBP_19`
- `OBP_20`
- `PA_16`
- `PA_17`
- `PA_18`
- `PA_19`
- `PA_20`
- Player age in 2021

The target variable is:

- `OBP_21`

## Results Shown

The page displays:

- Player name
- Predicted 2021 OBP
- Actual 2021 OBP
- Absolute error for each player
- Total absolute error
- Mean absolute error
- Number of players used to train the regression model

## Tech Stack

- HTML
- JavaScript
- PapaParse 5.3.0 from a CDN
- CSV data

## Repository Structure

```text
.
|-- index.htm    # Browser entry point and PapaParse CDN import
|-- script.js    # CSV parsing, linear regression model, and display updates
|-- obp.csv      # Player-level OBP and plate appearance data
|-- README.md    # Project documentation
`-- README.pdf   # PDF version of prior project documentation, if present
```

## How to Run the Project

Because the page uses `fetch("obp.csv")`, it should be served from a local web server instead of opened directly from the file system.

One simple option is to run a local server from the project directory:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/index.htm
```

The page will load the CSV, train the regression model, and display the predictions and error metrics.

## Notes and Limitations

- The model is trained and evaluated on the same dataset, so the displayed errors are useful for comparison but are not a true out-of-sample test.
- Missing values are handled with simple mean imputation from the training data.
- The regression is intentionally simple so the math can be read directly in the source code.
- The repository does not document the original source of the CSV data.
- The interface is minimal and focused on showing the model output clearly.
