document.addEventListener("DOMContentLoaded", function () {
  var MODEL_A_WEIGHT = 0.4;
  var MODEL_B_WEIGHT = 0.6;
  var TRADITIONAL_FEATURES = [
    "OBP_16",
    "OBP_17",
    "OBP_18",
    "OBP_19",
    "OBP_20",
    "PA_16",
    "PA_17",
    "PA_18",
    "PA_19",
    "PA_20",
  ];
  var HUMAN_FEATURES = [
    "height_inches",
    "weight_lbs",
    "age_2021",
    "region_east_weighted",
    "region_west_weighted",
    "region_foreign_weighted",
  ];
  var REGION_SOURCE_COLUMN = "hometown_region";
  var TARGET_COLUMN = "OBP_21";
  var DATA_FILE = "player_data.csv";
  var REGION_PRIOR_WEIGHTS = {
    region_central: 1.0,
    region_east: 1.15,
    region_west: 1.15,
    region_foreign: 1.3,
  };
  var MIN_OBP = 0;
  var MAX_OBP = 1;
  var SINGULARITY_EPSILON = 0.000000001;

  var table = document.querySelector("table");
  var tableHead = table ? table.querySelector("thead") : null;
  var predictionsTableBody = document.getElementById("predictions");
  var modelInfoDiv = document.getElementById("modelInfo");
  var totalErrorDiv = document.getElementById("totalError");
  var meanErrorDiv = document.getElementById("meanError");
  var summaryContainer = ensureSummaryContainer();

  initializeLayout();
  loadCsvData();

  function initializeLayout() {
    rewriteTableHeader();
    if (modelInfoDiv) {
      modelInfoDiv.textContent = "Loading two-stage OBP ensemble...";
    }
    if (totalErrorDiv) {
      totalErrorDiv.textContent = "Loading...";
    }
    if (meanErrorDiv) {
      meanErrorDiv.textContent = "Loading...";
    }
  }

  function loadCsvData() {
    Papa.parse(DATA_FILE, {
      download: true,
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: function (results) {
        try {
          var rawPlayers = results.data || [];
          validateSchema(rawPlayers);
          var normalizedPlayers = normalizePlayers(rawPlayers);
          var ensemblePackage = trainEnsemble(normalizedPlayers);
          renderResults(normalizedPlayers, ensemblePackage);
        } catch (error) {
          handleFailure(error);
        }
      },
      error: function (error) {
        handleFailure(error);
      },
    });
  }

  function validateSchema(players) {
    if (!players.length) {
      throw new Error("The CSV file is empty.");
    }

    var requiredColumns = TRADITIONAL_FEATURES.concat([
      "height_inches",
      "weight_lbs",
      REGION_SOURCE_COLUMN,
      TARGET_COLUMN,
    ]);
    var firstPlayer = players[0];
    var missingColumns = [];

    for (var i = 0; i < requiredColumns.length; i++) {
      if (
        !Object.prototype.hasOwnProperty.call(firstPlayer, requiredColumns[i])
      ) {
        missingColumns.push(requiredColumns[i]);
      }
    }

    if (
      !Object.prototype.hasOwnProperty.call(firstPlayer, "age_2021") &&
      !Object.prototype.hasOwnProperty.call(firstPlayer, "birth_date")
    ) {
      missingColumns.push("age_2021 (or birth_date fallback)");
    }

    if (missingColumns.length) {
      throw new Error(
        "Missing required CSV columns: " + missingColumns.join(", "),
      );
    }
  }

  function normalizePlayers(players) {
    return players.map(function (player) {
      var normalized = shallowCopy(player);
      var region = (normalized[REGION_SOURCE_COLUMN] || "")
        .toString()
        .trim()
        .toLowerCase();

      if (parseNumber(normalized.age_2021) === null && normalized.birth_date) {
        normalized.age_2021 = calculateAgeIn2021(normalized.birth_date);
      }

      normalized.region_east_weighted =
        region === "region_east" ? REGION_PRIOR_WEIGHTS.region_east : 0;
      normalized.region_west_weighted =
        region === "region_west" ? REGION_PRIOR_WEIGHTS.region_west : 0;
      normalized.region_foreign_weighted =
        region === "region_foreign" ? REGION_PRIOR_WEIGHTS.region_foreign : 0;
      normalized.region_central_weight = REGION_PRIOR_WEIGHTS.region_central;

      return normalized;
    });
  }

  function trainEnsemble(players) {
    var traditionalModel = trainLinearModel(
      players,
      TRADITIONAL_FEATURES,
      TARGET_COLUMN,
    );
    var humanModel = trainLinearModel(players, HUMAN_FEATURES, TARGET_COLUMN);

    if (!traditionalModel || !humanModel) {
      throw new Error(
        "The ensemble could not be trained because one of the feature groups does not have enough usable rows.",
      );
    }

    return {
      traditionalModel: traditionalModel,
      humanModel: humanModel,
    };
  }

  function trainLinearModel(players, featureNames, targetColumn) {
    var rows = [];
    var trainingPlayers = [];
    var i;

    for (i = 0; i < players.length; i++) {
      var row = extractFeatureRow(players[i], featureNames);
      var target = parseNumber(players[i][targetColumn]);

      if (target !== null) {
        rows.push(row);
        trainingPlayers.push({
          features: row,
          target: target,
        });
      }
    }

    if (trainingPlayers.length <= featureNames.length) {
      return null;
    }

    var imputationMeans = calculateColumnMeans(rows);
    var imputedRows = trainingPlayers.map(function (player) {
      return imputeMissingValues(player.features, imputationMeans);
    });
    var scaler = fitStandardScaler(imputedRows);
    var xMatrix = [];
    var yMatrix = [];

    for (i = 0; i < trainingPlayers.length; i++) {
      xMatrix.push([1].concat(transformWithScaler(imputedRows[i], scaler)));
      yMatrix.push([trainingPlayers[i].target]);
    }

    return {
      featureNames: featureNames.slice(),
      coefficients: solveNormalEquation(xMatrix, yMatrix),
      imputationMeans: imputationMeans,
      scaler: scaler,
      trainingRowCount: trainingPlayers.length,
    };
  }

  function renderResults(players, ensemblePackage) {
    if (!predictionsTableBody) {
      return;
    }

    predictionsTableBody.innerHTML = "";

    var modelA = ensemblePackage.traditionalModel;
    var modelB = ensemblePackage.humanModel;
    var predictionRows = [];
    var traditionalActualPairs = [];
    var blendedActualPairs = [];

    for (var i = 0; i < players.length; i++) {
      var player = players[i];
      var actual = parseNumber(player[TARGET_COLUMN]);
      var modelAPrediction = predictWithModel(player, modelA);
      var modelBPrediction = predictWithModel(player, modelB);
      var blendedPrediction =
        modelAPrediction !== null && modelBPrediction !== null
          ? clamp(
              MODEL_A_WEIGHT * modelAPrediction +
                MODEL_B_WEIGHT * modelBPrediction,
              MIN_OBP,
              MAX_OBP,
            )
          : null;

      if (actual !== null && modelAPrediction !== null) {
        traditionalActualPairs.push({
          actual: actual,
          prediction: modelAPrediction,
        });
      }

      if (actual !== null && blendedPrediction !== null) {
        blendedActualPairs.push({
          actual: actual,
          prediction: blendedPrediction,
        });
      }

      predictionRows.push({
        name: player.Name || "Unknown Player",
        actual: actual,
        modelAPrediction: modelAPrediction,
        modelBPrediction: modelBPrediction,
        blendedPrediction: blendedPrediction,
      });
    }

    predictionRows.sort(function (left, right) {
      return (
        numericSortValue(absoluteError(left.blendedPrediction, left.actual)) -
        numericSortValue(absoluteError(right.blendedPrediction, right.actual))
      );
    });

    for (var rowIndex = 0; rowIndex < predictionRows.length; rowIndex++) {
      appendPredictionRow(predictionRows[rowIndex]);
    }

    updateSummaryView({
      modelA: modelA,
      modelB: modelB,
      modelAMae: calculateMeanAbsoluteError(traditionalActualPairs),
      blendedMae: calculateMeanAbsoluteError(blendedActualPairs),
      blendedCoverage: blendedActualPairs.length,
    });
  }

  function appendPredictionRow(predictionRow) {
    var row = document.createElement("tr");
    addTableCell(row, predictionRow.name);
    addTableCell(row, formatObp(predictionRow.modelAPrediction));
    addTableCell(row, formatObp(predictionRow.modelBPrediction));
    addTableCell(row, formatObp(predictionRow.blendedPrediction));
    addTableCell(row, formatObp(predictionRow.actual));
    addTableCell(
      row,
      formatError(
        absoluteError(predictionRow.modelAPrediction, predictionRow.actual),
      ),
    );
    addTableCell(
      row,
      formatError(
        absoluteError(predictionRow.blendedPrediction, predictionRow.actual),
      ),
    );
    predictionsTableBody.appendChild(row);
  }

  function updateSummaryView(summary) {
    if (modelInfoDiv) {
      modelInfoDiv.textContent =
        "Two-stage ensemble trained in pure JavaScript. Model A uses only historical OBP/PA, Model B uses biometrics plus weighted regional indicators derived from hometown labels, and the final projection blends them at a fixed 40/60 stats-to-human split. Regional priors are set so foreign players carry the largest preset signal, east and west coast players are next, and central players are the reference group.";
    }

    if (totalErrorDiv) {
      totalErrorDiv.textContent =
        summary.blendedMae === null
          ? "N/A"
          : (summary.blendedMae * summary.blendedCoverage).toFixed(3);
    }

    if (meanErrorDiv) {
      meanErrorDiv.textContent =
        summary.blendedMae === null ? "N/A" : summary.blendedMae.toFixed(3);
    }

    if (!summaryContainer) {
      return;
    }

    summaryContainer.innerHTML = "";
    summaryContainer.appendChild(
      createMetricBlock(
        "Training Rows (Model A)",
        String(summary.modelA.trainingRowCount),
      ),
    );
    summaryContainer.appendChild(
      createMetricBlock(
        "Training Rows (Model B)",
        String(summary.modelB.trainingRowCount),
      ),
    );
    summaryContainer.appendChild(
      createMetricBlock("Model A MAE", formatError(summary.modelAMae)),
    );
    summaryContainer.appendChild(
      createMetricBlock(
        "Blended Ensemble MAE",
        formatError(summary.blendedMae),
      ),
    );
    summaryContainer.appendChild(
      createMetricBlock(
        "Ensemble Weight Split",
        "0.40 traditional / 0.60 human",
      ),
    );
    summaryContainer.appendChild(
      createMetricBlock(
        "Regional Prior Weights",
        "foreign 1.30 | east 1.15 | west 1.15 | central 1.00",
      ),
    );
    summaryContainer.appendChild(
      createMetricBlock(
        "Human Coefficients",
        formatCoefficientSummary(
          summary.modelB.featureNames,
          summary.modelB.coefficients,
        ),
      ),
    );
  }

  function rewriteTableHeader() {
    if (!tableHead) {
      return;
    }

    tableHead.innerHTML = "";

    var headerRow = document.createElement("tr");
    var headers = [
      "Player",
      "Model A OBP",
      "Model B OBP",
      "Blended OBP",
      "Actual OBP",
      "Model A Abs Error",
      "Blended Abs Error",
    ];

    for (var i = 0; i < headers.length; i++) {
      var cell = document.createElement("th");
      cell.textContent = headers[i];
      headerRow.appendChild(cell);
    }

    tableHead.appendChild(headerRow);
  }

  function ensureSummaryContainer() {
    if (!table || !table.parentNode) {
      return null;
    }

    var existing = document.getElementById("benchmarkSummary");
    if (existing) {
      return existing;
    }

    var container = document.createElement("div");
    container.id = "benchmarkSummary";
    container.style.display = "grid";
    container.style.gridTemplateColumns =
      "repeat(auto-fit, minmax(220px, 1fr))";
    container.style.gap = "12px";
    container.style.margin = "18px 0";
    table.parentNode.insertBefore(container, table);
    return container;
  }

  function createMetricBlock(label, value) {
    var block = document.createElement("div");
    block.style.border = "1px solid #cccccc";
    block.style.padding = "12px";
    block.style.backgroundColor = "#fafafa";

    var heading = document.createElement("div");
    heading.style.fontWeight = "bold";
    heading.style.marginBottom = "6px";
    heading.textContent = label;

    var text = document.createElement("div");
    text.textContent = value;

    block.appendChild(heading);
    block.appendChild(text);
    return block;
  }

  function extractFeatureRow(player, featureNames) {
    return featureNames.map(function (featureName) {
      return parseNumber(player[featureName]);
    });
  }

  function calculateColumnMeans(rows) {
    var sums = new Array(rows[0].length).fill(0);
    var counts = new Array(rows[0].length).fill(0);

    for (var i = 0; i < rows.length; i++) {
      for (var j = 0; j < rows[i].length; j++) {
        if (rows[i][j] !== null) {
          sums[j] += rows[i][j];
          counts[j] += 1;
        }
      }
    }

    return sums.map(function (sum, index) {
      return counts[index] > 0 ? sum / counts[index] : 0;
    });
  }

  function imputeMissingValues(row, means) {
    return row.map(function (value, index) {
      return value === null ? means[index] : value;
    });
  }

  function fitStandardScaler(rows) {
    var columnCount = rows[0].length;
    var means = new Array(columnCount).fill(0);
    var standardDeviations = new Array(columnCount).fill(0);
    var i;
    var j;

    for (i = 0; i < rows.length; i++) {
      for (j = 0; j < columnCount; j++) {
        means[j] += rows[i][j];
      }
    }

    for (j = 0; j < columnCount; j++) {
      means[j] = means[j] / rows.length;
    }

    for (i = 0; i < rows.length; i++) {
      for (j = 0; j < columnCount; j++) {
        standardDeviations[j] += Math.pow(rows[i][j] - means[j], 2);
      }
    }

    for (j = 0; j < columnCount; j++) {
      standardDeviations[j] =
        Math.sqrt(standardDeviations[j] / rows.length) || 1;
    }

    return {
      means: means,
      standardDeviations: standardDeviations,
    };
  }

  function transformWithScaler(row, scaler) {
    return row.map(function (value, index) {
      return (value - scaler.means[index]) / scaler.standardDeviations[index];
    });
  }

  function predictWithModel(player, model) {
    if (!model) {
      return null;
    }

    var rawFeatureRow = extractFeatureRow(player, model.featureNames);
    var imputedRow = imputeMissingValues(rawFeatureRow, model.imputationMeans);
    var scaledRow = transformWithScaler(imputedRow, model.scaler);
    var designRow = [1].concat(scaledRow);
    var output = multiplyMatrices([designRow], model.coefficients);

    if (!output.length || !output[0].length || !isFinite(output[0][0])) {
      return null;
    }

    return clamp(output[0][0], MIN_OBP, MAX_OBP);
  }

  function solveNormalEquation(xMatrix, yMatrix) {
    var xTranspose = transposeMatrix(xMatrix);
    var xtx = multiplyMatrices(xTranspose, xMatrix);
    var determinantValue = determinant(xtx);

    if (Math.abs(determinantValue) < SINGULARITY_EPSILON) {
      xtx = addDiagonalJitter(xtx, SINGULARITY_EPSILON);
    }

    return multiplyMatrices(
      invertMatrix(xtx),
      multiplyMatrices(xTranspose, yMatrix),
    );
  }

  function transposeMatrix(matrix) {
    var transposed = [];

    for (var column = 0; column < matrix[0].length; column++) {
      transposed[column] = [];
      for (var row = 0; row < matrix.length; row++) {
        transposed[column][row] = matrix[row][column];
      }
    }

    return transposed;
  }

  function multiplyMatrices(left, right) {
    var leftColumns = left[0].length;
    var rightRows = right.length;

    if (leftColumns !== rightRows) {
      throw new Error("Matrix multiplication dimension mismatch.");
    }

    var result = [];

    for (var row = 0; row < left.length; row++) {
      result[row] = [];
      for (var column = 0; column < right[0].length; column++) {
        var sum = 0;
        for (var inner = 0; inner < leftColumns; inner++) {
          sum += left[row][inner] * right[inner][column];
        }
        result[row][column] = sum;
      }
    }

    return result;
  }

  function determinant(matrix) {
    if (matrix.length !== matrix[0].length) {
      throw new Error("Determinant requires a square matrix.");
    }

    var working = cloneMatrix(matrix);
    var size = working.length;
    var determinantValue = 1;
    var sign = 1;

    for (var pivotIndex = 0; pivotIndex < size; pivotIndex++) {
      var pivotRow = pivotIndex;

      for (var row = pivotIndex + 1; row < size; row++) {
        if (
          Math.abs(working[row][pivotIndex]) >
          Math.abs(working[pivotRow][pivotIndex])
        ) {
          pivotRow = row;
        }
      }

      if (Math.abs(working[pivotRow][pivotIndex]) < SINGULARITY_EPSILON) {
        return 0;
      }

      if (pivotRow !== pivotIndex) {
        var temp = working[pivotIndex];
        working[pivotIndex] = working[pivotRow];
        working[pivotRow] = temp;
        sign *= -1;
      }

      var pivot = working[pivotIndex][pivotIndex];
      determinantValue *= pivot;

      for (row = pivotIndex + 1; row < size; row++) {
        var factor = working[row][pivotIndex] / pivot;
        for (var column = pivotIndex; column < size; column++) {
          working[row][column] -= factor * working[pivotIndex][column];
        }
      }
    }

    return determinantValue * sign;
  }

  function invertMatrix(matrix) {
    if (matrix.length !== matrix[0].length) {
      throw new Error("Matrix inversion requires a square matrix.");
    }

    var size = matrix.length;
    var augmented = [];
    var row;
    var column;

    for (row = 0; row < size; row++) {
      augmented[row] = [];
      for (column = 0; column < size; column++) {
        augmented[row][column] = matrix[row][column];
      }
      for (column = 0; column < size; column++) {
        augmented[row][column + size] = row === column ? 1 : 0;
      }
    }

    for (column = 0; column < size; column++) {
      var pivotRow = column;

      for (row = column + 1; row < size; row++) {
        if (
          Math.abs(augmented[row][column]) >
          Math.abs(augmented[pivotRow][column])
        ) {
          pivotRow = row;
        }
      }

      if (Math.abs(augmented[pivotRow][column]) < SINGULARITY_EPSILON) {
        throw new Error("Matrix is singular and cannot be inverted.");
      }

      if (pivotRow !== column) {
        var tempRow = augmented[column];
        augmented[column] = augmented[pivotRow];
        augmented[pivotRow] = tempRow;
      }

      var pivot = augmented[column][column];

      for (
        var normalizedColumn = 0;
        normalizedColumn < size * 2;
        normalizedColumn++
      ) {
        augmented[column][normalizedColumn] /= pivot;
      }

      for (row = 0; row < size; row++) {
        if (row === column) {
          continue;
        }

        var factor = augmented[row][column];
        for (
          var eliminationColumn = 0;
          eliminationColumn < size * 2;
          eliminationColumn++
        ) {
          augmented[row][eliminationColumn] -=
            factor * augmented[column][eliminationColumn];
        }
      }
    }

    var inverse = [];
    for (row = 0; row < size; row++) {
      inverse[row] = augmented[row].slice(size);
    }

    return inverse;
  }

  function addDiagonalJitter(matrix, epsilon) {
    var stabilized = cloneMatrix(matrix);

    for (var i = 0; i < stabilized.length; i++) {
      stabilized[i][i] += epsilon;
    }

    return stabilized;
  }

  function cloneMatrix(matrix) {
    return matrix.map(function (row) {
      return row.slice();
    });
  }

  function calculateMeanAbsoluteError(pairs) {
    if (!pairs.length) {
      return null;
    }

    var total = 0;

    for (var i = 0; i < pairs.length; i++) {
      total += Math.abs(pairs[i].prediction - pairs[i].actual);
    }

    return total / pairs.length;
  }

  function absoluteError(prediction, actual) {
    if (prediction === null || actual === null) {
      return null;
    }

    return Math.abs(prediction - actual);
  }

  function addTableCell(row, text) {
    var cell = document.createElement("td");
    cell.textContent = text;
    row.appendChild(cell);
  }

  function formatObp(value) {
    return value === null ? "N/A" : value.toFixed(3);
  }

  function formatError(value) {
    return value === null ? "N/A" : value.toFixed(3);
  }

  function formatCoefficientSummary(featureNames, coefficientMatrix) {
    var parts = [];

    for (var i = 0; i < featureNames.length; i++) {
      parts.push(
        featureNames[i] + ": " + coefficientMatrix[i + 1][0].toFixed(4),
      );
    }

    return parts.join(" | ");
  }

  function numericSortValue(value) {
    return value === null ? Number.POSITIVE_INFINITY : value;
  }

  function calculateAgeIn2021(birthDateText) {
    var birthDate = new Date(birthDateText);

    if (isNaN(birthDate.getTime())) {
      return null;
    }

    var seasonReferenceDate = new Date("2021-07-01T00:00:00");
    var age = seasonReferenceDate.getFullYear() - birthDate.getFullYear();
    var hadBirthday =
      seasonReferenceDate.getMonth() > birthDate.getMonth() ||
      (seasonReferenceDate.getMonth() === birthDate.getMonth() &&
        seasonReferenceDate.getDate() >= birthDate.getDate());

    return hadBirthday ? age : age - 1;
  }

  function parseNumber(value) {
    if (value === null || value === undefined || value === "") {
      return null;
    }

    var numericValue = parseFloat(value);
    return isNaN(numericValue) ? null : numericValue;
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function shallowCopy(object) {
    var copy = {};
    for (var key in object) {
      if (Object.prototype.hasOwnProperty.call(object, key)) {
        copy[key] = object[key];
      }
    }
    return copy;
  }

  function handleFailure(error) {
    console.error(error);

    if (modelInfoDiv) {
      modelInfoDiv.textContent =
        "Model failed to load. Check the CSV schema and browser console for details.";
    }

    if (totalErrorDiv) {
      totalErrorDiv.textContent = "N/A";
    }

    if (meanErrorDiv) {
      meanErrorDiv.textContent = "N/A";
    }

    if (summaryContainer) {
      summaryContainer.innerHTML = "";
      summaryContainer.appendChild(
        createMetricBlock(
          "Load Error",
          error && error.message ? error.message : String(error),
        ),
      );
    }
  }
});
