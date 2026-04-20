document.addEventListener("DOMContentLoaded", function () {
  // David Shableski
  // dbshableski@gmail.com
  // Updated regression version
  //
  // This project uses OBP and plate appearances from 2016-2020, plus the
  // player's age in 2021, to train a simple linear regression model that
  // predicts OBP for the 2021 season.

  var predictionsTableBody = document.getElementById("predictions");
  var totalErrorDiv = document.getElementById("totalError");
  var meanErrorDiv = document.getElementById("meanError");
  var modelInfoDiv = document.getElementById("modelInfo");

  var featureColumns = [
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
    "age_2021",
  ];

  fetch("obp.csv")
    .then(function (response) {
      return response.text();
    })
    .then(function (csvText) {
      Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
        complete: function (results) {
          var players = results.data;
          var model = trainLinearRegression(players);
          displayPredictions(players, model);
        },
        error: function (error) {
          console.error("Error parsing CSV:", error);
        },
      });
    })
    .catch(function (error) {
      console.error("Error fetching the CSV file:", error);
    });

  function displayPredictions(players, model) {
    predictionsTableBody.innerHTML = "";

    if (!model) {
      modelInfoDiv.textContent = "The model could not be trained because there were not enough usable rows.";
      totalErrorDiv.textContent = "N/A";
      meanErrorDiv.textContent = "N/A";
      return;
    }

    var totalAbsoluteError = 0;
    var errorCount = 0;

    for (var i = 0; i < players.length; i++) {
      var player = players[i];
      var row = document.createElement("tr");
      var actualOBP = parseNumber(player["OBP_21"]);
      var prediction = predictPlayerOBP(player, model);

      var error = null;
      if (prediction !== null && actualOBP !== null) {
        error = Math.abs(prediction - actualOBP);
        totalAbsoluteError += error;
        errorCount++;
      }

      addTableCell(row, player["Name"] || "Unknown Player");
      addTableCell(row, prediction === null ? "N/A" : prediction.toFixed(3));
      addTableCell(row, actualOBP === null ? "N/A" : actualOBP.toFixed(3));
      addTableCell(row, error === null ? "N/A" : error.toFixed(3));
      predictionsTableBody.appendChild(row);
    }

    var meanAbsoluteError = errorCount > 0 ? totalAbsoluteError / errorCount : 0;

    modelInfoDiv.textContent =
      "Linear regression trained on " +
      model.trainingRowCount +
      " players using historical OBP, plate appearances, and age.";
    totalErrorDiv.textContent = totalAbsoluteError.toFixed(3);
    meanErrorDiv.textContent = meanAbsoluteError.toFixed(3);
  }

  function addTableCell(row, text) {
    var cell = document.createElement("td");
    cell.textContent = text;
    row.appendChild(cell);
  }

  function trainLinearRegression(players) {
    // First, collect rows with a real 2021 OBP target and enough history to
    // make a useful training example.
    var candidateRows = [];

    for (var i = 0; i < players.length; i++) {
      var trainingRow = buildFeatureRow(players[i]);
      var target = parseNumber(players[i]["OBP_21"]);

      if (target !== null && trainingRow.validSeasonCount >= 2 && trainingRow.age !== null) {
        candidateRows.push({
          features: trainingRow.features,
          target: target,
        });
      }
    }

    if (candidateRows.length < featureColumns.length + 1) {
      return null;
    }

    var imputeMeans = calculateFeatureMeans(candidateRows);
    var filledFeatureRows = [];

    for (var j = 0; j < candidateRows.length; j++) {
      filledFeatureRows.push(fillMissingFeatures(candidateRows[j].features, imputeMeans));
    }

    var scaleInfo = calculateScaleInfo(filledFeatureRows);
    var xMatrix = [];
    var yValues = [];

    for (var k = 0; k < candidateRows.length; k++) {
      xMatrix.push([1].concat(standardizeFeatures(filledFeatureRows[k], scaleInfo)));
      yValues.push(candidateRows[k].target);
    }

    // Linear regression is fit with the normal equation:
    // coefficients = inverse(X'X) X'y.
    // A small ridge value is added to keep the equation stable if columns are
    // closely related or a feature has very little variation.
    var coefficients = solveLinearRegression(xMatrix, yValues, 0.01);

    return {
      coefficients: coefficients,
      imputeMeans: imputeMeans,
      scaleInfo: scaleInfo,
      trainingRowCount: candidateRows.length,
    };
  }

  function buildFeatureRow(player) {
    var features = [];
    var validSeasonCount = 0;

    for (var i = 0; i < featureColumns.length; i++) {
      if (featureColumns[i] === "age_2021") {
        var age = calculateAgeIn2021(player["birth_date"]);
        features.push(age);
      } else {
        features.push(parseNumber(player[featureColumns[i]]));
      }
    }

    for (var year = 16; year <= 20; year++) {
      if (parseNumber(player["OBP_" + year]) !== null && parseNumber(player["PA_" + year]) !== null) {
        validSeasonCount++;
      }
    }

    return {
      features: features,
      validSeasonCount: validSeasonCount,
      age: calculateAgeIn2021(player["birth_date"]),
    };
  }

  function predictPlayerOBP(player, model) {
    var featureRow = buildFeatureRow(player);

    if (featureRow.validSeasonCount < 1 || featureRow.age === null) {
      return null;
    }

    var filledFeatures = fillMissingFeatures(featureRow.features, model.imputeMeans);
    var standardizedFeatures = standardizeFeatures(filledFeatures, model.scaleInfo);
    var modelInputs = [1].concat(standardizedFeatures);
    var prediction = 0;

    for (var i = 0; i < model.coefficients.length; i++) {
      prediction += model.coefficients[i] * modelInputs[i];
    }

    // OBP is a rate, so keep unusual regression outputs inside a realistic range.
    return clamp(prediction, 0, 1);
  }

  function calculateFeatureMeans(rows) {
    var sums = new Array(featureColumns.length).fill(0);
    var counts = new Array(featureColumns.length).fill(0);

    for (var i = 0; i < rows.length; i++) {
      for (var j = 0; j < featureColumns.length; j++) {
        if (rows[i].features[j] !== null) {
          sums[j] += rows[i].features[j];
          counts[j]++;
        }
      }
    }

    return sums.map(function (sum, index) {
      return counts[index] > 0 ? sum / counts[index] : 0;
    });
  }

  function fillMissingFeatures(features, imputeMeans) {
    return features.map(function (value, index) {
      return value === null ? imputeMeans[index] : value;
    });
  }

  function calculateScaleInfo(featureRows) {
    var means = new Array(featureColumns.length).fill(0);
    var standardDeviations = new Array(featureColumns.length).fill(0);

    for (var i = 0; i < featureRows.length; i++) {
      for (var j = 0; j < featureColumns.length; j++) {
        means[j] += featureRows[i][j];
      }
    }

    for (var m = 0; m < means.length; m++) {
      means[m] = means[m] / featureRows.length;
    }

    for (var row = 0; row < featureRows.length; row++) {
      for (var col = 0; col < featureColumns.length; col++) {
        standardDeviations[col] += Math.pow(featureRows[row][col] - means[col], 2);
      }
    }

    for (var s = 0; s < standardDeviations.length; s++) {
      standardDeviations[s] = Math.sqrt(standardDeviations[s] / featureRows.length) || 1;
    }

    return {
      means: means,
      standardDeviations: standardDeviations,
    };
  }

  function standardizeFeatures(features, scaleInfo) {
    return features.map(function (value, index) {
      return (value - scaleInfo.means[index]) / scaleInfo.standardDeviations[index];
    });
  }

  function solveLinearRegression(xMatrix, yValues, ridgeValue) {
    var transposedX = transposeMatrix(xMatrix);
    var xtx = multiplyMatrices(transposedX, xMatrix);
    var xty = multiplyMatrixAndVector(transposedX, yValues);

    for (var i = 1; i < xtx.length; i++) {
      xtx[i][i] += ridgeValue;
    }

    return solveLinearSystem(xtx, xty);
  }

  function transposeMatrix(matrix) {
    var transposed = [];

    for (var col = 0; col < matrix[0].length; col++) {
      transposed[col] = [];
      for (var row = 0; row < matrix.length; row++) {
        transposed[col][row] = matrix[row][col];
      }
    }

    return transposed;
  }

  function multiplyMatrices(a, b) {
    var result = [];

    for (var row = 0; row < a.length; row++) {
      result[row] = [];
      for (var col = 0; col < b[0].length; col++) {
        var sum = 0;
        for (var inner = 0; inner < b.length; inner++) {
          sum += a[row][inner] * b[inner][col];
        }
        result[row][col] = sum;
      }
    }

    return result;
  }

  function multiplyMatrixAndVector(matrix, vector) {
    var result = [];

    for (var row = 0; row < matrix.length; row++) {
      var sum = 0;
      for (var col = 0; col < vector.length; col++) {
        sum += matrix[row][col] * vector[col];
      }
      result[row] = sum;
    }

    return result;
  }

  function solveLinearSystem(matrix, vector) {
    // Gaussian elimination with partial pivoting solves the small system used
    // by this project without needing an external math or machine learning library.
    var size = matrix.length;
    var augmented = [];

    for (var i = 0; i < size; i++) {
      augmented[i] = matrix[i].slice();
      augmented[i].push(vector[i]);
    }

    for (var col = 0; col < size; col++) {
      var pivotRow = col;

      for (var row = col + 1; row < size; row++) {
        if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivotRow][col])) {
          pivotRow = row;
        }
      }

      if (pivotRow !== col) {
        var temp = augmented[col];
        augmented[col] = augmented[pivotRow];
        augmented[pivotRow] = temp;
      }

      var pivot = augmented[col][col];
      if (Math.abs(pivot) < 0.000000001) {
        pivot = 0.000000001;
      }

      for (var nextRow = col + 1; nextRow < size; nextRow++) {
        var factor = augmented[nextRow][col] / pivot;

        for (var nextCol = col; nextCol <= size; nextCol++) {
          augmented[nextRow][nextCol] -= factor * augmented[col][nextCol];
        }
      }
    }

    var solution = new Array(size).fill(0);

    for (var backRow = size - 1; backRow >= 0; backRow--) {
      var sum = augmented[backRow][size];

      for (var backCol = backRow + 1; backCol < size; backCol++) {
        sum -= augmented[backRow][backCol] * solution[backCol];
      }

      solution[backRow] = sum / augmented[backRow][backRow];
    }

    return solution;
  }

  function calculateAgeIn2021(birthDateText) {
    var birthDate = new Date(birthDateText);

    if (isNaN(birthDate.getTime())) {
      return null;
    }

    var age = 2021 - birthDate.getFullYear();
    var birthdayPassed =
      birthDate.getMonth() < 6 || (birthDate.getMonth() === 6 && birthDate.getDate() <= 1);

    return birthdayPassed ? age : age - 1;
  }

  function parseNumber(value) {
    if (value === null || value === undefined || value === "") {
      return null;
    }

    var number = parseFloat(value);
    return isNaN(number) ? null : number;
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }
});
