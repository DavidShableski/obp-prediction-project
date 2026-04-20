# MLB On-Base Percentage Projection Tool

## Project Summary

This project is a browser-based analytical tool for estimating MLB player on-base percentage (OBP) for the 2021 season using historical OBP and plate appearance data from 2016 through 2020. The tool parses a local CSV file, calculates a weighted historical projection for each player, compares the estimate with actual 2021 OBP, and displays the total absolute error across the dataset.

## Project Overview / Problem Statement

The project addresses a practical sports analytics problem: using player performance records from the 2016-2020 MLB seasons to form a transparent projection for a 2021 season outcome. Rather than using a black-box model, the implementation uses a rules-based weighted average so the calculation can be inspected directly in the JavaScript source.

The projection gives more influence to more recent seasons and to seasons with more plate appearances. It also applies a simple age adjustment for players older than 30. The output is intended to make the prediction logic and resulting error visible for each player in the dataset.

## Dataset / Inputs Used

The primary input is [`obp.csv`](obp.csv), a local CSV file containing 572 player records. The columns include:

- Player identifiers: `Name`, `playerid`, `birth_date`
- Actual 2021 values: `PA_21`, `OBP_21`
- Historical inputs: `PA_20`, `OBP_20`, `PA_19`, `OBP_19`, `PA_18`, `OBP_18`, `PA_17`, `OBP_17`, `PA_16`, `OBP_16`

The repository does not document the original source of the CSV data, so the README does not make a claim about data provenance.

## Methodology / Approach

The calculation is implemented in [`script.js`](script.js):

1. The browser fetches `obp.csv`.
2. PapaParse parses the CSV with the first row treated as headers.
3. For each player, the script reads OBP and plate appearance values from 2016 through 2020.
4. Each season receives a recency multiplier:
   - 2016: `PA * 1`
   - 2017: `PA * 2`
   - 2018: `PA * 3`
   - 2019: `PA * 4`
   - 2020: `PA * 5`
5. Seasons with missing or zero OBP are assigned zero weight.
6. The projected 2021 OBP is calculated as a weighted average of historical OBP values.
7. Players older than 30, based on `birth_date`, receive a 5% reduction to the projected OBP.
8. The script compares projected 2021 OBP with actual `OBP_21` and adds the absolute error to a total error value.

This is a heuristic projection method, not a trained machine learning model.

## Key Features

- Parses local baseball performance data directly in the browser.
- Uses plate appearances and recency to weight historical OBP values.
- Handles missing historical OBP values by excluding those seasons from the weighted calculation.
- Applies a simple age-based adjustment for players older than 30.
- Displays each player's projected 2021 OBP alongside actual 2021 OBP.
- Calculates and displays the total absolute error across all parsed player records with valid projections.

## Tech Stack

- HTML for the browser page structure.
- JavaScript for data loading, projection logic, DOM updates, and error calculation.
- PapaParse 5.3.0, loaded from a CDN, for CSV parsing.
- CSV as the local data input format.

## Repository Structure

```text
.
|-- index.htm    # Browser entry point and PapaParse CDN import
|-- script.js    # CSV parsing, OBP projection logic, and display updates
|-- obp.csv      # Player-level OBP and plate appearance data
|-- README.md    # Project documentation
`-- README.pdf   # PDF version of prior project documentation
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

The page will load `obp.csv`, calculate the projections, and display the total absolute error and player-level results.

## Notes / Limitations

- The projection is based on a hand-coded weighted average, not a fitted statistical or machine learning model.
- The age adjustment is a fixed 5% reduction for players older than 30 and is not estimated from the dataset.
- The repository does not include tests, package configuration, or a build process.
- The data source for `obp.csv` is not documented in the repository.
- The interface is minimal and focused on displaying calculation output rather than interactive analysis.
- Some inline comments in `script.js` refer to batting average, but the implemented calculation uses OBP columns.

## Future Improvements

- Document the source and collection process for the CSV data.
- Add validation for missing or malformed numeric fields before calculating projections.
- Report additional error metrics, such as mean absolute error, to make model evaluation easier to interpret.
- Separate calculation logic from DOM rendering so the projection method can be tested independently.
- Add tests for the weighting, missing-data handling, age adjustment, and error calculation.
- Improve the results display with sorting, filtering, and clearer formatting for player-level comparisons.
